# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from pytorch3d.loss import chamfer_distance, mesh_edge_loss
from pytorch3d.ops import GraphConv, sample_points_from_meshes, vert_align
from pytorch3d.structures import Meshes
from torch import nn
from torch.nn import functional as F
from meshrcnn.utils.chamfer import chamfer_distance2
from scipy.spatial import cKDTree
from meshrcnn.structures.mesh import MeshInstances, batch_crop_meshes_within_box

ROI_MESH_HEAD_REGISTRY = Registry("ROI_MESH_HEAD")
binary_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')


def mesh_rcnn_loss(
    pred_meshes,
    instances,
    loss_weights=None,
    gt_num_samples=5000,
    pred_num_samples=5000,
    gt_coord_thresh=None,
):
    """
    Compute the mesh prediction loss defined in the Mesh R-CNN paper.

    Args:
        pred_meshes (list of Meshes): A list of K Meshes. Each entry contains B meshes,
            where B is the total number of predicted meshes in all images.
            K is the number of refinements
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1 correspondence with the pred_meshes.
            The ground-truth labels (class, box, mask, ...) associated with each instance
            are stored in fields.
        loss_weights (dict): Contains the weights for the different losses, e.g.
            loss_weights = {'champfer': 1.0, 'normals': 0.0, 'edge': 0.2}
        gt_num_samples (int): The number of points to sample from gt meshes
        pred_num_samples (int): The number of points to sample from predicted meshes
        gt_coord_thresh (float): A threshold value over which the batch is ignored
    Returns:
        mesh_loss (Tensor): A scalar tensor containing the loss.
    """
    if not isinstance(pred_meshes, list):
        raise ValueError("Expecting a list of Meshes")

    gt_verts, gt_faces = [], []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        gt_K = instances_per_image.gt_K
        gt_mesh_per_image = batch_crop_meshes_within_box(
            instances_per_image.gt_meshes, instances_per_image.proposal_boxes.tensor, gt_K
        ).to(device=pred_meshes[0].device)
        gt_verts.extend(gt_mesh_per_image.verts_list())
        gt_faces.extend(gt_mesh_per_image.faces_list())

    if len(gt_verts) == 0:
        return None, None

    gt_meshes = Meshes(verts=gt_verts, faces=gt_faces)
    gt_valid = gt_meshes.valid
    gt_sampled_verts, gt_sampled_normals = sample_points_from_meshes(
        gt_meshes, num_samples=gt_num_samples, return_normals=True
    )

    all_loss_chamfer = []
    all_loss_normals = []
    all_loss_edge = []
    for pred_mesh in pred_meshes:
        pred_sampled_verts, pred_sampled_normals = sample_points_from_meshes(
            pred_mesh, num_samples=pred_num_samples, return_normals=True
        )
        wts = (pred_mesh.valid * gt_valid).to(dtype=torch.float32)
        # chamfer loss
        loss_chamfer, loss_normals = chamfer_distance(
            pred_sampled_verts,
            gt_sampled_verts,
            x_normals=pred_sampled_normals,
            y_normals=gt_sampled_normals,
            weights=wts,
        )
        # chamfer loss
        loss_chamfer = loss_chamfer * loss_weights["chamfer"]
        all_loss_chamfer.append(loss_chamfer)
        # normal loss
        loss_normals = loss_normals * loss_weights["normals"]
        all_loss_normals.append(loss_normals)
        # mesh edge regularization
        loss_edge = mesh_edge_loss(pred_mesh)
        loss_edge = loss_edge * loss_weights["edge"]
        all_loss_edge.append(loss_edge)
    loss_chamfer = sum(all_loss_chamfer)
    loss_normals = sum(all_loss_normals)
    loss_edge = sum(all_loss_edge)

    # if the rois are bad, the target verts can be arbitrarily large
    # causing exploding gradients. If this is the case, ignore the batch
    if gt_coord_thresh and gt_sampled_verts.abs().max() > gt_coord_thresh:
        loss_chamfer = loss_chamfer * 0.0
        loss_normals = loss_normals * 0.0
        loss_edge = loss_edge * 0.0

    return loss_chamfer, loss_normals, loss_edge, gt_meshes

# From Total3DUnderstanding
def SVRLoss(
    pred_meshes: list[Meshes],
    points_from_edges,
    point_indicators,
    instances,
    loss_weights=None,
    gt_coord_thresh=None,
):

    if not isinstance(pred_meshes, list):
        raise ValueError("Expecting a list of Meshes")

    gt_verts, gt_faces, gt_verts_density = [], [], []
    device = pred_meshes[0].device
    # Iterate over batch (in our case 1)
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        verts = torch.stack([mesh[0] for mesh in instances_per_image.gt_meshes], dim=0)
        faces = [mesh[1] for mesh in instances_per_image.gt_meshes]
        gt_verts.append(verts)
        gt_faces.append(faces)
        gt_verts_density.append(instances_per_image.gt_densities)


    if len(gt_verts) == 0:
        return None, None
    gt_verts = torch.vstack(gt_verts)  # (N, V, 3)
    gt_verts_density = torch.vstack(gt_verts_density)    # (N, V)

    loss_chamfer = torch.tensor(0.).to(device)
    loss_edge = torch.tensor(0.).to(device)
    loss_face = torch.tensor(0.).to(device)
    loss_boundary = torch.tensor(0.).to(device)

    for stage, pred_mesh in enumerate(pred_meshes):
        # chamfer loss
        dist1, dist2 = chamfer_distance2(gt_verts, torch.stack(pred_mesh.verts_list()))[:2]
        loss_chamfer += ((torch.mean(dist1)) + (torch.mean(dist2))) * loss_weights["chamfer"]
        # edge regularization
        loss_edge_aux = mesh_edge_loss(pred_mesh)
        loss_edge += loss_edge_aux * loss_weights["edge"]
        # boundary loss
        # if stage == len(pred_meshes) - 1:
        #     pass
    # face distance losses
    for points_from_edges_by_step, points_indicator_by_step in zip(points_from_edges,
                                                                   point_indicators):
        points_from_edges_by_step = points_from_edges_by_step.transpose(1, 2).contiguous()  # (N, 3, E) -> (N, E, 3)
        _, dist2_face, _, _, _, idx2 = chamfer_distance2(gt_verts, points_from_edges_by_step)   # (N, E), (N, E, 1)
        local_dens = torch.gather(gt_verts_density, 1, idx2.squeeze(-1))
        in_mesh = (dist2_face <= local_dens).float()
        loss_face += binary_cls_criterion(points_indicator_by_step, in_mesh) * loss_weights["face"]

    loss_chamfer = 100 * loss_chamfer / len(pred_meshes)
    loss_edge = 100 * loss_edge / len(pred_meshes)
    loss_boundary = 100 * loss_boundary
    if points_from_edges:
        loss_face = loss_face / len(points_from_edges)

    # if the rois are bad, the target verts can be arbitrarily large
    # causing exploding gradients. If this is the case, ignore the batch
    if gt_coord_thresh and gt_verts.abs().max() > gt_coord_thresh:
        print("asdasd")
        loss_chamfer = loss_chamfer * 0.0
        loss_face = loss_face * 0.0
        loss_edge = loss_edge * 0.0
        loss_boundary = loss_boundary * 0.0

    return loss_chamfer, loss_edge, loss_face, loss_boundary, Meshes(verts=[], faces=[])


def mesh_rcnn_inference(pred_meshes, pred_instances):
    """
    Return the predicted mesh for each predicted instance

    Args:
        pred_meshes (Meshes): A class of Meshes containing B meshes, where B is
            the total number of predictions in all images.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_meshes" field storing the meshes
    """
    num_boxes_per_image = [len(i) for i in pred_instances]
    pred_meshes = pred_meshes.split(num_boxes_per_image)

    for pred_mesh, instances in zip(pred_meshes, pred_instances):
        # NOTE do not save the Meshes object; pickle dumps become inefficient
        if pred_mesh.isempty():
            continue
        verts_list = pred_mesh.verts_list()
        faces_list = pred_mesh.faces_list()
        instances.pred_meshes = MeshInstances([(v, f) for (v, f) in zip(verts_list, faces_list)])


class MeshRefinementStage(nn.Module):
    def __init__(self, img_feat_dim, vert_feat_dim, hidden_dim, stage_depth, gconv_init="normal"):
        """
        Args:
          img_feat_dim: Dimension of features we will get from vert_align
          vert_feat_dim: Dimension of vert_feats we will receive from the
                        previous stage; can be 0
          hidden_dim: Output dimension for graph-conv layers
          stage_depth: Number of graph-conv layers to use
          gconv_init: How to initialize graph-conv layers
        """
        super(MeshRefinementStage, self).__init__()

        # fc layer to reduce feature dimension
        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)

        # deform layer
        self.verts_offset = nn.Linear(hidden_dim + 3, 3)

        # graph convs
        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3
            else:
                input_dim = hidden_dim + 3
            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs.append(gconv)

        # initialization
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)

        nn.init.zeros_(self.verts_offset.weight)
        nn.init.constant_(self.verts_offset.bias, 0)

    def forward(self, x, mesh, vert_feats=None):
        # print('vert_align x: ',x.shape)
        img_feats = vert_align(x, mesh, return_packed=True, padding_mode="border") # turn to 2 dim
        # 256 -> hidden_dim
        # print(img_feats.shape)  # N X 256
        img_feats = F.relu(self.bottleneck(img_feats))
        # print('img_feats: ',img_feats.shape)  # N X hid
        if vert_feats is None:
            # hidden_dim + 3 from start procedure
            vert_feats = torch.cat((img_feats, mesh.verts_packed()), dim=1)
            # print('first layer:', vert_feats.shape)
        else:
            # hidden_dim * 2 + 3
            vert_feats = torch.cat((vert_feats, img_feats, mesh.verts_packed()), dim=1)
            # print('not first layer:', vert_feats.shape)
        for graph_conv in self.gconvs:
            vert_feats_nopos = F.relu(graph_conv(vert_feats, mesh.edges_packed()))
            vert_feats = torch.cat((vert_feats_nopos, mesh.verts_packed()), dim=1)
            # print('graph vert_feats_nopos: ', vert_feats_nopos.shape)
            # print('graph: ', vert_feats.shape)
        # refine
        deform = torch.tanh(self.verts_offset(vert_feats))
        # print('deform: ', deform.shape)
        # print('mesh: ', mesh.verts_packed().shape)
        mesh = mesh.offset_verts(deform)
        # print('mesh: ', mesh.shape)
        return mesh, vert_feats_nopos

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 1024, output_dim = 3):
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)
        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class EREstimate(nn.Module):
    def __init__(self, bottleneck_size=1024, output_dim = 1):
        super(EREstimate, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)

        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


@ROI_MESH_HEAD_REGISTRY.register()
class MeshRCNNGraphConvHead(nn.Module):
    """
    A mesh head with vert align, graph conv layers and refine layers.
    """
    def __init__(self, cfg, input_shape: ShapeSpec): # [256, 14, 14, None]
        super(MeshRCNNGraphConvHead, self).__init__()
        num_stages         = cfg.MODEL.ROI_MESH_HEAD.NUM_STAGES
        num_graph_convs    = cfg.MODEL.ROI_MESH_HEAD.NUM_GRAPH_CONVS  # per stage
        graph_conv_dim     = cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_DIM
        graph_conv_init    = cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_INIT
        input_channels     = input_shape.channels

        self.use_atlas = not cfg.MODEL.VOXEL_ON
        if self.use_atlas:
            self.edge_classifier_on = True
            self.face_samples = 1
            self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            self.subnetworks = num_stages
            self.bottleneck_size = 1024
            self.avPooling = nn.MaxPool2d((input_shape.height,input_shape.width))
            self.encoder = nn.Linear(input_shape.channels, self.bottleneck_size)
            self.decoders = nn.ModuleList([PointGenCon(bottleneck_size= self.bottleneck_size+self.num_classes+3) for i in range(0, self.subnetworks)])
            if self.edge_classifier_on:
                self.edge_classifiers = nn.ModuleList([EREstimate(bottleneck_size=self.bottleneck_size+self.num_classes+3, output_dim=1) for i in range(0, max(self.subnetworks - 1, 1))])
                self.remove_edge_th = 0.1
        else:
            self.stages = nn.ModuleList()
            for i in range(num_stages):
                vert_feat_dim = 0 if i == 0 else graph_conv_dim
                stage = MeshRefinementStage(
                    input_channels,
                    vert_feat_dim,
                    graph_conv_dim,
                    num_graph_convs,
                    gconv_init=graph_conv_init,
                )
                self.stages.append(stage)

    def forward(self, x, mesh:Meshes, classes):
        """

        Args:
            x: proposals feature map (N, channels, h, w)
            mesh: initial sphere
            classes: class information of the proposals

        Returns:
            train: Meshes, points_from_edges (in case of Atlas),
            test: Meshes

        """
        if x.numel() == 0 or mesh.isempty():
            return [Meshes(verts=[], faces=[])]
        meshes = []
        # ATLAS
        if self.use_atlas:
            device = mesh.device
            N = x.shape[0]
            V = mesh.num_verts_per_mesh()[0].cpu()
            E = mesh.num_edges_per_mesh()[0].cpu()

            points_from_edges = []  # list of each stage of mesh_sampled_points
            point_indicators = []  # list of each stage of estimated distance to gt (output of edge classifier)
            boundary_point_ids = torch.zeros(size=(N, V), dtype=torch.uint8).to(device)  # (N, V)
            remove_edges_list = []  # list of each stage removed edges

            verts = mesh.verts_list()[0].repeat(N, 1, 1).permute(0, 2, 1)  # (N, 3, V)
            edges = mesh.edges_packed().repeat(N, 1, 1)  # (N, E, 2)
            faces = mesh.faces_list()[0].repeat(N, 1, 1) # (N, F, 3)
            face_to_edges = mesh.faces_packed_to_edges_packed()

            oneHot_classes = nn.functional.one_hot(classes, self.num_classes)   # (N, C)
            flattened_x = torch.flatten(self.avPooling(x), start_dim=1) # (N,256)
            im_latent = self.encoder(flattened_x) # (N, 1024)
            im_latent = torch.cat((im_latent, oneHot_classes), dim=1) # (N, 1024+C)
            im_latent_V_expanded = im_latent.unsqueeze(2).expand(im_latent.shape[0], im_latent.shape[1], V)   # (N, 1024+C, V)
            im_latent_E_expanded = im_latent.unsqueeze(2).expand(im_latent.shape[0], im_latent.shape[1], E)   # (N, 1024+C, E)

            for i in range(self.subnetworks):
                # DEFORMING VERTICES
                verts_with_latent = torch.cat((im_latent_V_expanded, verts), dim=1)   # (N, 1024+C+3, V)
                verts = verts + self.decoders[i](verts_with_latent) # (N, 3, V)
                # EDGE PRUNING
                if self.edge_classifier_on:
                    if i < self.subnetworks - 1:
                        edge_sampled_points = sample_points_on_edges(verts, edges, quantity=self.face_samples, training=self.training) # (N, 3, E)
                        points_from_edges.append(edge_sampled_points)
                        edges_with_latent = torch.cat((im_latent_E_expanded, edge_sampled_points), dim=1)   # (N, 1024+C+3, E)
                        indicators = self.edge_classifiers[i](edges_with_latent)    # (N, 1, E)
                        indicators = indicators.view(N, 1, E, self.face_samples)    # (N, 1, E, FS)
                        indicators = indicators.squeeze(1)  # (N, E, FS)
                        indicators = torch.mean(indicators, dim=2) # (N, E)
                        point_indicators.append(indicators)
                        remove_edges = torch.nonzero(torch.sigmoid(indicators) < self.remove_edge_th)   # [ [N_idx, E_idx] ]
                        remove_edges_list.append(remove_edges)
                        if not self.training:
                            new_faces = []
                        for batch_id in range(N):
                            rm_edges = remove_edges[remove_edges[:, 0] == batch_id, 1]
                            if len(rm_edges) > 0:
                                edges[batch_id][rm_edges, :] = 0    # (setting edges to [0,0])
                            if not self.training:
                                new_batch_faces = remove_faces_according_to_edge(faces[batch_id], edges[batch_id], face_to_edges)
                                new_faces.append(new_batch_faces)
                    # else:
                    #     remove_edges_list = [item for item in remove_edges_list if len(item)]
                    #     if remove_edges_list:
                    #         remove_edges_list = torch.unique(torch.cat(remove_edges_list), dim=0)
                    #         for batch_id in range(N):
                    #             rm_edges = remove_edges_list[remove_edges_list[:, 0] == batch_id, 1]
                    #             if len(rm_edges) > 0:
                    #                 rm_candidates, counts = torch.unique(sphere_edges[rm_edges], return_counts=True)
                    #                 boundary_ids = counts < sphere_adjacency[rm_candidates - 1].sum(1)
                    #                 boundary_point_ids[batch_id][rm_candidates[boundary_ids] - 1] = 1

                meshes.append(Meshes(verts=list(verts.permute(0,2,1)),
                                     faces=new_faces if self.edge_classifier_on and not self.training else list(faces)))
            return meshes, points_from_edges, point_indicators
        # MESHRCNN
        else:
            vert_feats = None
            for stage in self.stages:
                mesh, vert_feats = stage(x, mesh, vert_feats=vert_feats)
                meshes.append(mesh)
            return meshes

def sample_points_on_edges(points, edges, quantity = 1, training=True):
    n_batch = edges.shape[0]
    n_edges = edges.shape[1]

    if training:
        weights = np.diff(np.sort(np.vstack(
            [np.zeros((1, n_edges * quantity)), np.random.uniform(0, 1, size=(1, n_edges * quantity)),
             np.ones((1, n_edges * quantity))]), axis=0), axis=0)
    else:
        weights = 0.5 * np.ones((2, n_edges * quantity))

    weights = weights.reshape([2, quantity, n_edges])
    weights = torch.from_numpy(weights).float().to(points.device)
    weights = weights.transpose(1, 2)
    weights = weights.transpose(0, 1).contiguous()
    weights = weights.expand(n_batch, n_edges, 2, quantity).contiguous()
    weights = weights.view(n_batch * n_edges, 2, quantity)


    left_nodes = torch.gather(points.transpose(1, 2), 1, (edges[:, :, 0]).unsqueeze(-1).expand(edges.size(0), edges.size(1), 3))
    right_nodes = torch.gather(points.transpose(1, 2), 1, (edges[:, :, 1]).unsqueeze(-1).expand(edges.size(0), edges.size(1), 3))
    edge_points = torch.cat([left_nodes.unsqueeze(-1), right_nodes.unsqueeze(-1)], -1).view(n_batch*n_edges, 3, 2)
    new_point_set = torch.bmm(edge_points, weights).contiguous()
    new_point_set = new_point_set.view(n_batch, n_edges, 3, quantity)
    new_point_set = new_point_set.transpose(2, 3).contiguous()
    new_point_set = new_point_set.view(n_batch, n_edges * quantity, 3)
    new_point_set = new_point_set.transpose(1, 2).contiguous()
    return new_point_set

def remove_faces_according_to_edge(faces, edges, face_to_edge):
    remaining_idx = torch.where(torch.any(edges != 0, axis=1))[0]   # idx of edges with any node != 0 ie to remain
    to_remain_mask = torch.isin(face_to_edge, remaining_idx)    # either each of the edges of the face is to remain or not
    to_remain_idx = torch.where(torch.all(to_remain_mask, axis=1))[0]   # idx of faces where all 3 edges are to be reamined
    return faces[to_remain_idx,:]

def build_mesh_head(cfg, input_shape):
    name = cfg.MODEL.ROI_MESH_HEAD.NAME
    return ROI_MESH_HEAD_REGISTRY.get(name)(cfg, input_shape)
