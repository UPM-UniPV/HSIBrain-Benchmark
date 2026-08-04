import math
import torch

import torch.nn.functional as F


def geodesic_separability(
	x: torch.Tensor,
	labels: torch.Tensor,
	k: int = 50,
	eps: float = 1e-8,
	inf: float = 1e9
):
	"""
	Geodesic separability metric based on:

	Multi-manifold Discriminant Isomap for visualization and classification
	https://www.sciencedirect.com/science/article/pii/S0031320316000571

	
	"""

	device = x.device
	N = x.shape[0]

	x = x.detach()
	labels = labels.detach()

	mean_f = x.mean(dim=0, keepdims=True)
	std_f = x.std(dim=0, keepdims=True)+eps
	x = (x-mean_f)/std_f

	dists = torch.cdist(x, x)
	
	knn_dists, knn_idx = torch.topk(
		dists,
		k=k + 1,
		largest=False
	)

	# remove self neighbor
	knn_dists = knn_dists[:, 1:]
	knn_idx = knn_idx[:, 1:]

	graph = torch.full((N, N), inf, device=device)

	row_idx = (
		torch.arange(N, device=device)
		.unsqueeze(1)
		.expand(-1, k)
	)

	graph[row_idx, knn_idx] = knn_dists
	graph = torch.minimum(graph, graph.T)
	graph.fill_diagonal_(0)

	for m in range(N):
		graph = torch.minimum(
			graph,
			graph[:, m:m+1] + graph[m:m+1, :]
		)

	shortest = graph

	labels = labels.view(-1)
	same = labels[:, None] == labels[None, :]
	diff = ~same

	same.fill_diagonal_(False)

	same_dists = shortest.clone()
	same_dists[~same] = inf

	d_same = same_dists.min(dim=1).values

	diff_dists = shortest.clone()
	diff_dists[~diff] = inf

	d_diff = diff_dists.min(dim=1).values

	margins = (d_diff - d_same) / (
		d_diff + d_same + eps
	)

	return margins.mean().cpu().numpy()


def compute_si_r1(feature_vectors, labels, dist_met="euc", eps=1e-8):
	"""
	Computes the Separation Index (SI) for r=1 using Euclidean distance.

	Args:
		feature_vectors (Tensor): shape (N, F) flattened feature vectors
		labels (Tensor): shape (N,) integer class labels

	Returns:
		float: SI value in [0, 1]
	"""
	feature_vectors = feature_vectors.detach()
	labels = labels.detach()

	if dist_met == "euc":
		mean_f = feature_vectors.mean(dim=0, keepdims=True)
		std_f = feature_vectors.std(dim=0, keepdims=True)+eps
		feature_vectors = (feature_vectors-mean_f)/std_f
		distance_matrix = torch.cdist(feature_vectors, feature_vectors, p=2)
	
	elif dist_met == "cos":
		feature_vectors_norm = F.normalize(feature_vectors, p=2, dim=1)
		distance_matrix = 1 - (feature_vectors_norm @ feature_vectors_norm.T)


	distance_matrix.fill_diagonal_(float("inf"))

	nearest_idx = torch.argmin(distance_matrix, dim=1)
	correct = (labels == labels[nearest_idx]).float()

	return round(correct.mean().item(), 5)



def compute_gdv(feature_vectors, labels, dist_met="euc", eps=1e-8):

	feature_vectors = feature_vectors.detach()
	labels = labels.detach()

	""" Z-SCORE IN FEATS """

	if dist_met == "euc":
		mean_f = feature_vectors.mean(dim=0, keepdims=True)
		std_f = feature_vectors.std(dim=0, keepdims=True)+eps
		feature_vectors = .5*(feature_vectors-mean_f)/std_f
		distance_matrix = torch.cdist(feature_vectors, feature_vectors, p=2)
	
	elif dist_met == "cos":
		feature_vectors_norm = F.normalize(feature_vectors, p=2, dim=1)
		distance_matrix = 1 - (feature_vectors_norm @ feature_vectors_norm.T)


	_, D = feature_vectors.shape

	unique_classes, _ = torch.sort(labels.unique())

	L = len(unique_classes)


	labels_row = labels.unsqueeze(1)  # (N, 1)
	labels_col = labels.unsqueeze(0)  # (1, N)

	inter_c_dist = []
	intra_c_dist = []

	for l in unique_classes:
		for m in unique_classes:

			if m<l:
				continue # already computed metrics
			
			target_mask = 1*((labels_row==l) & (labels_col==m))
			target_mask = torch.triu(target_mask, diagonal=1)

			if l==m:
				# Intra-class distance computation
				N_l = torch.sum(target_mask)
				
				d = torch.sum(distance_matrix[target_mask>0])
				intra_dist = 2*d/(N_l*(N_l-1))
				intra_c_dist.append(intra_dist)
				
			else:
				# Inter-class distance computation
				N_l = torch.sum(labels==l)
				N_m = torch.sum(labels==m)

				d = torch.sum(distance_matrix[target_mask>0])
				inter_dist = d/(N_l*N_m)
				inter_c_dist.append(inter_dist)
	
	intra_c_dist = torch.FloatTensor(intra_c_dist)
	inter_c_dist = torch.FloatTensor(inter_c_dist)

	gdv = (torch.sum(intra_c_dist)/L - 2*torch.sum(inter_c_dist)/(L*(L-1)))/math.sqrt(D)
	return gdv


def compute_sep(X, Y, metric="GEO"):

		
	if len(X.shape)==5:
		B,C,D,H,W = X.shape
		X = X.view(B, C*D, H, W)
	
	elif len(X.shape)==4:
		B,C,H,W = X.shape
	
	if len(X.shape)>3:
		if H>1:
			X = X[...,H//2, W//2]

	X = X.squeeze()

	if len(X.shape)==3:
		B,E,C = X.shape
		X = X.view(B, C*E)

	if metric == "SI":
		sep_ix = compute_si_r1(X.squeeze(), Y)
	
	elif metric == "GDV":
		sep_ix = compute_gdv(X.squeeze(), Y)

	elif metric == "GEO":
		sep_ix = geodesic_separability(X.squeeze(), Y)


	return sep_ix
