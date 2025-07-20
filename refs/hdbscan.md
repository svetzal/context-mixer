# API Reference[¶](https://hdbscan.readthedocs.io/en/latest/api.html#api-reference "Link to this heading")

Major classes are `HDBSCAN` and `RobustSingleLinkage`.

## HDBSCAN[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan "Link to this heading")

_class_hdbscan.hdbscan_.HDBSCAN(_min_cluster_size=5_, _min_samples=None_, _cluster_selection_epsilon=0.0_, _cluster_selection_persistence=0.0_, _max_cluster_size=0_, _metric='euclidean'_, _alpha=1.0_, _p=None_, _algorithm='best'_, _leaf_size=40_, _memory=Memory(location=None)_, _approx_min_span_tree=True_, _gen_min_span_tree=False_, _core_dist_n_jobs=4_, _cluster_selection_method='eom'_, _allow_single_cluster=False_, _prediction_data=False_, _branch_detection_data=False_, _match_reference_implementation=False_, _cluster_selection_epsilon_max=inf_, _**kwargs_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/hdbscan_.html#HDBSCAN)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN "Link to this definition")

Perform HDBSCAN clustering from vector array or distance matrix.

HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications with Noise. Performs DBSCAN over varying epsilon values and integrates the result to find a clustering that gives the best stability over epsilon. This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN), and be more robust to parameter selection.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#parameters "Link to this heading")

min_cluster_sizeint, optional (default=5)

The minimum size of clusters; single linkage splits that contain fewer points than this will be considered points “falling out” of a cluster rather than a cluster splitting into two new clusters.

min_samplesint, optional (default=None)

The number of samples in a neighborhood for a point to be considered as a core point. This includes the point itself. When None, defaults to min_cluster_size.

metricstring, or callable, optional (default=’euclidean’)

The metric to use when calculating distance between instances in a feature array. If metric is a string or callable, it must be one of the options allowed by metrics.pairwise.pairwise_distances for its metric parameter. If metric is “precomputed”, X is assumed to be a distance matrix and must be square.

pint, optional (default=None)

p value to use if using the minkowski metric.

alphafloat, optional (default=1.0)

A distance scaling parameter as used in robust single linkage. See [3](https://hdbscan.readthedocs.io/en/latest/api.html#id7) for more information.

cluster_selection_epsilon: float, optional (default=0.0)

A distance threshold. Clusters below this value will be merged. This is the minimum epsilon allowed. See [5](https://hdbscan.readthedocs.io/en/latest/api.html#id9) for more information.

cluster_selection_persistence: float, optional (default=0.0)

A persistence threshold. Clusters with a persistence lower than this value will be merged.

algorithmstring, optional (default=’best’)

Exactly which algorithm to use; hdbscan has variants specialised for different characteristics of the data. By default this is set to `best` which chooses the “best” algorithm given the nature of the data. You can force other options if you believe you know better. Options are:

> - `best`
>     
> - `generic`
>     
> - `prims_kdtree`
>     
> - `prims_balltree`
>     
> - `boruvka_kdtree`
>     
> - `boruvka_balltree`
>     

leaf_size: int, optional (default=40)

If using a space tree algorithm (kdtree, or balltree) the number of points ina leaf node of the tree. This does not alter the resulting clustering, but may have an effect on the runtime of the algorithm.

memoryInstance of joblib.Memory or string (optional)

Used to cache the output of the computation of the tree. By default, no caching is done. If a string is given, it is the path to the caching directory.

approx_min_span_treebool, optional (default=True)

Whether to accept an only approximate minimum spanning tree. For some algorithms this can provide a significant speedup, but the resulting clustering may be of marginally lower quality. If you are willing to sacrifice speed for correctness you may want to explore this; in general this should be left at the default True.

gen_min_span_tree: bool, optional (default=False)

Whether to generate the minimum spanning tree with regard to mutual reachability distance for later analysis.

core_dist_n_jobsint, optional (default=4)

Number of parallel jobs to run in core distance computations (if supported by the specific algorithm). For `core_dist_n_jobs` below -1, (n_cpus + 1 + core_dist_n_jobs) are used.

cluster_selection_methodstring, optional (default=’eom’)

The method used to select clusters from the condensed tree. The standard approach for HDBSCAN* is to use an Excess of Mass algorithm to find the most persistent clusters. Alternatively you can instead select the clusters at the leaves of the tree – this provides the most fine grained and homogeneous clusters. Options are:

> - `eom`
>     
> - `leaf`
>     

allow_single_clusterbool, optional (default=False)

By default HDBSCAN* will not produce a single cluster, setting this to True will override this and allow single cluster results in the case that you feel this is a valid result for your dataset.

prediction_databoolean, optional

Whether to generate extra cached data for predicting labels or membership vectors for new unseen points later. If you wish to persist the clustering object for later re-use you probably want to set this to True. (default False)

branch_detection_databoolean, optional

Whether to generated extra cached data for detecting branch- hierarchies within clusters. If you wish to use functions from `hdbscan.branches` set this to True. (default False)

match_reference_implementationbool, optional (default=False)

There exist some interpretational differences between this HDBSCAN* implementation and the original authors reference implementation in Java. This can result in very minor differences in clustering results. Setting this flag to True will, at a some performance cost, ensure that the clustering results match the reference implementation.

cluster_selection_epsilon_max: float, optional (default=inf)

A distance threshold. Clusters above this value will be split. Has no effect when using leaf clustering (where clusters are usually small regardless) and can also be overridden in rare cases by a high value for cluster_selection_epsilon. Note that this should not be used if we want to predict the cluster labels for new points in future (e.g. using approximate_predict), as the approximate_predict function is not aware of this argument. This is the maximum epsilon allowed.

[**](https://hdbscan.readthedocs.io/en/latest/api.html#id3)kwargsoptional

Arguments passed to the distance metric

### Attributes[¶](https://hdbscan.readthedocs.io/en/latest/api.html#attributes "Link to this heading")

[labels_](https://hdbscan.readthedocs.io/en/latest/api.html#id106)ndarray, shape (n_samples, )

Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.

[probabilities_](https://hdbscan.readthedocs.io/en/latest/api.html#id108)ndarray, shape (n_samples, )

The strength with which each sample is a member of its assigned cluster. Noise points have probability zero; points in clusters have values assigned proportional to the degree that they persist as part of the cluster.

[cluster_persistence_](https://hdbscan.readthedocs.io/en/latest/api.html#id110)ndarray, shape (n_clusters, )

A score of how persistent each cluster is. A score of 1.0 represents a perfectly stable cluster that persists over all distance scales, while a score of 0.0 represents a perfectly ephemeral cluster. These scores can be gauge the relative coherence of the clusters output by the algorithm.

[condensed_tree_](https://hdbscan.readthedocs.io/en/latest/api.html#id112)CondensedTree object

The condensed tree produced by HDBSCAN. The object has methods for converting to pandas, networkx, and plotting.

[single_linkage_tree_](https://hdbscan.readthedocs.io/en/latest/api.html#id114)SingleLinkageTree object

The single linkage tree produced by HDBSCAN. The object has methods for converting to pandas, networkx, and plotting.

[minimum_spanning_tree_](https://hdbscan.readthedocs.io/en/latest/api.html#id116)MinimumSpanningTree object

The minimum spanning tree of the mutual reachability graph generated by HDBSCAN. Note that this is not generated by default and will only be available if gen_min_span_tree was set to True on object creation. Even then in some optimized cases a tre may not be generated.

[outlier_scores_](https://hdbscan.readthedocs.io/en/latest/api.html#id118)ndarray, shape (n_samples, )

Outlier scores for clustered points; the larger the score the more outlier-like the point. Useful as an outlier detection technique. Based on the GLOSH algorithm by Campello, Moulavi, Zimek and Sander.

[prediction_data_](https://hdbscan.readthedocs.io/en/latest/api.html#id120)PredictionData object

Cached data used for predicting the cluster labels of new or unseen points. Necessary only if you are using functions from `hdbscan.prediction` (see [`approximate_predict()`](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.prediction.approximate_predict "hdbscan.prediction.approximate_predict"),[`membership_vector()`](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.prediction.membership_vector "hdbscan.prediction.membership_vector"), and [`all_points_membership_vectors()`](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.prediction.all_points_membership_vectors "hdbscan.prediction.all_points_membership_vectors")).

[branch_detection_data_](https://hdbscan.readthedocs.io/en/latest/api.html#id122)BranchDetectionData object

Cached data used for detecting branch-hierarchies within clusters. Necessary only if you are using function from `hdbscan.branches`.

[exemplars_](https://hdbscan.readthedocs.io/en/latest/api.html#id124)list

A list of exemplar points for clusters. Since HDBSCAN supports arbitrary shapes for clusters we cannot provide a single cluster exemplar per cluster. Instead a list is returned with each element of the list being a numpy array of exemplar points for a cluster – these points are the “most representative” points of the cluster.

[relative_validity_](https://hdbscan.readthedocs.io/en/latest/api.html#id126)float

A fast approximation of the Density Based Cluster Validity (DBCV) score [4]. The only difference, and the speed, comes from the fact that this [relative_validity_](https://hdbscan.readthedocs.io/en/latest/api.html#id128) is computed using the mutual- reachability minimum spanning tree, i.e. [minimum_spanning_tree_](https://hdbscan.readthedocs.io/en/latest/api.html#id130), instead of the all-points minimum spanning tree used in the reference. This score might not be an objective measure of the goodness of clustering. It may only be used to compare results across different choices of hyper-parameters, therefore is only a relative score.

### References[¶](https://hdbscan.readthedocs.io/en/latest/api.html#references "Link to this heading")

[1]

Campello, R. J., Moulavi, D., & Sander, J. (2013, April). Density-based clustering based on hierarchical density estimates. In Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp. 160-172). Springer Berlin Heidelberg.

[2]

Campello, R. J., Moulavi, D., Zimek, A., & Sander, J. (2015). Hierarchical density estimates for data clustering, visualization, and outlier detection. ACM Transactions on Knowledge Discovery from Data (TKDD), 10(1), 5.

[3]([1](https://hdbscan.readthedocs.io/en/latest/api.html#id1),[2](https://hdbscan.readthedocs.io/en/latest/api.html#id94),[3](https://hdbscan.readthedocs.io/en/latest/api.html#id95))

Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the cluster tree. In Advances in Neural Information Processing Systems (pp. 343-351).

[4]

Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J., 2014. Density-Based Clustering Validation. In SDM (pp. 839-847).

[[5](https://hdbscan.readthedocs.io/en/latest/api.html#id2)]

Malzer, C., & Baum, M. (2019). A Hybrid Approach To Hierarchical Density-based Cluster Selection. arxiv preprint 1911.02282.

dbscan_clustering(_cut_distance_, _min_cluster_size=5_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/hdbscan_.html#HDBSCAN.dbscan_clustering)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN.dbscan_clustering "Link to this definition")

Return clustering that would be equivalent to running DBSCAN* for a particular cut_distance (or epsilon) DBSCAN* can be thought of as DBSCAN without the border points. As such these results may differ slightly from sklearns implementation of dbscan in the non-core points.

This can also be thought of as a flat clustering derived from constant height cut through the single linkage tree.

This represents the result of selecting a cut value for robust single linkage clustering. The min_cluster_size allows the flat clustering to declare noise points (and cluster smaller than min_cluster_size).

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id10 "Link to this heading")

cut_distancefloat

The mutual reachability distance cut value to use to generate a flat clustering.

min_cluster_sizeint, optional

Clusters smaller than this value with be called ‘noise’ and remain unclustered in the resulting flat clustering.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#returns "Link to this heading")

labelsarray [n_samples]

An array of cluster labels, one per datapoint. Unclustered points are assigned the label -1.

fit(_X_, _y=None_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/hdbscan_.html#HDBSCAN.fit)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN.fit "Link to this definition")

Perform HDBSCAN clustering from features or distance matrix.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id11 "Link to this heading")

Xarray or sparse (CSR) matrix of shape (n_samples, n_features), or array of shape (n_samples, n_samples)

A feature array, or array of distances between samples if `metric='precomputed'`.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id12 "Link to this heading")

selfobject

Returns self

fit_predict(_X_, _y=None_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/hdbscan_.html#HDBSCAN.fit_predict)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN.fit_predict "Link to this definition")

Performs clustering on X and returns cluster labels.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id13 "Link to this heading")

Xarray or sparse (CSR) matrix of shape (n_samples, n_features), or array of shape (n_samples, n_samples)

A feature array, or array of distances between samples if `metric='precomputed'`.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id14 "Link to this heading")

yndarray, shape (n_samples, )

cluster labels

generate_branch_detection_data()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/hdbscan_.html#HDBSCAN.generate_branch_detection_data)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN.generate_branch_detection_data "Link to this definition")

Create data that caches intermediate results used for detecting branches within clusters. This data is only useful if you are intending to use functions from `hdbscan.branches`.

generate_prediction_data()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/hdbscan_.html#HDBSCAN.generate_prediction_data)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN.generate_prediction_data "Link to this definition")

Create data that caches intermediate results used for predicting the label of new/unseen points. This data is only useful if you are intending to use functions from `hdbscan.prediction`.

weighted_cluster_centroid(_cluster_id_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/hdbscan_.html#HDBSCAN.weighted_cluster_centroid)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN.weighted_cluster_centroid "Link to this definition")

Provide an approximate representative point for a given cluster. Note that this technique assumes a euclidean metric for speed of computation. For more general metrics use the `weighted_cluster_medoid` method which is slower, but can work with the metric the model trained with.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id15 "Link to this heading")

cluster_id: int

The id of the cluster to compute a centroid for.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id16 "Link to this heading")

centroid: array of shape (n_features,)

A representative centroid for cluster `cluster_id`.

weighted_cluster_medoid(_cluster_id_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/hdbscan_.html#HDBSCAN.weighted_cluster_medoid)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.hdbscan_.HDBSCAN.weighted_cluster_medoid "Link to this definition")

Provide an approximate representative point for a given cluster. Note that this technique can be very slow and memory intensive for large clusters. For faster results use the `weighted_cluster_centroid` method which is faster, but assumes a euclidean metric.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id17 "Link to this heading")

cluster_id: int

The id of the cluster to compute a medoid for.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id18 "Link to this heading")

centroid: array of shape (n_features,)

A representative medoid for cluster `cluster_id`.

## RobustSingleLinkage[¶](https://hdbscan.readthedocs.io/en/latest/api.html#robustsinglelinkage "Link to this heading")

_class_hdbscan.robust_single_linkage_.RobustSingleLinkage(_cut=0.4_, _k=5_, _alpha=1.4142135623730951_, _gamma=5_, _metric='euclidean'_, _algorithm='best'_, _core_dist_n_jobs=4_, _metric_params={}_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/robust_single_linkage_.html#RobustSingleLinkage)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.robust_single_linkage_.RobustSingleLinkage "Link to this definition")

Perform robust single linkage clustering from a vector array or distance matrix.

Robust single linkage is a modified version of single linkage that attempts to be more robust to noise. Specifically the goal is to more accurately approximate the level set tree of the unknown probability density function from which the sample data has been drawn.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id19 "Link to this heading")

Xarray or sparse (CSR) matrix of shape (n_samples, n_features), or 

> array of shape (n_samples, n_samples)

A feature array, or array of distances between samples if `metric='precomputed'`.

cutfloat

The reachability distance value to cut the cluster heirarchy at to derive a flat cluster labelling.

kint, optional (default=5)

Reachability distances will be computed with regard to the k nearest neighbors.

alphafloat, optional (default=np.sqrt(2))

Distance scaling for reachability distance computation. Reachability distance is computed as $max { core_k(a), core_k(b), 1/alpha d(a,b) }$.

gammaint, optional (default=5)

Ignore any clusters in the flat clustering with size less than gamma, and declare points in such clusters as noise points.

metricstring, or callable, optional (default=’euclidean’)

The metric to use when calculating distance between instances in a feature array. If metric is a string or callable, it must be one of the options allowed by metrics.pairwise.pairwise_distances for its metric parameter. If metric is “precomputed”, X is assumed to be a distance matrix and must be square.

metric_paramsdict, option (default={})

Keyword parameter arguments for calling the metric (for example the p values if using the minkowski metric).

algorithmstring, optional (default=’best’)

Exactly which algorithm to use; hdbscan has variants specialised for different characteristics of the data. By default this is set to `best` which chooses the “best” algorithm given the nature of the data. You can force other options if you believe you know better. Options are:

> - `small`
>     
> - `small_kdtree`
>     
> - `large_kdtree`
>     
> - `large_kdtree_fastcluster`
>     

core_dist_n_jobsint, optional

Number of parallel jobs to run in core distance computations (if supported by the specific algorithm). For `core_dist_n_jobs` below -1, (n_cpus + 1 + core_dist_n_jobs) are used. (default 4)

### Attributes[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id20 "Link to this heading")

[labels_](https://hdbscan.readthedocs.io/en/latest/api.html#id132)ndarray, shape (n_samples, )

Cluster labels for each point. Noisy samples are given the label -1.

[cluster_hierarchy_](https://hdbscan.readthedocs.io/en/latest/api.html#id134)SingleLinkageTree object

The single linkage tree produced during clustering. This object provides several methods for:

> - Plotting
>     
> - Generating a flat clustering
>     
> - Exporting to NetworkX
>     
> - Exporting to Pandas
>     

### References[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id21 "Link to this heading")

[1]

Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the cluster tree. In Advances in Neural Information Processing Systems (pp. 343-351).

fit(_X_, _y=None_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/robust_single_linkage_.html#RobustSingleLinkage.fit)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.robust_single_linkage_.RobustSingleLinkage.fit "Link to this definition")

Perform robust single linkage clustering from features or distance matrix.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id23 "Link to this heading")

Xarray or sparse (CSR) matrix of shape (n_samples, n_features), or array of shape (n_samples, n_samples)

A feature array, or array of distances between samples if `metric='precomputed'`.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id24 "Link to this heading")

selfobject

Returns self

fit_predict(_X_, _y=None_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/robust_single_linkage_.html#RobustSingleLinkage.fit_predict)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.robust_single_linkage_.RobustSingleLinkage.fit_predict "Link to this definition")

Performs clustering on X and returns cluster labels.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id25 "Link to this heading")

Xarray or sparse (CSR) matrix of shape (n_samples, n_features), or array of shape (n_samples, n_samples)

A feature array, or array of distances between samples if `metric='precomputed'`.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id26 "Link to this heading")

yndarray, shape (n_samples, )

cluster labels

## Utilities[¶](https://hdbscan.readthedocs.io/en/latest/api.html#utilities "Link to this heading")

Other useful classes are contained in the plots module, the validity module, and the prediction module.

_class_hdbscan.plots.CondensedTree(_condensed_tree_array_, _labels_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#CondensedTree)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.CondensedTree "Link to this definition")

The condensed tree structure, which provides a simplified or smoothed version of the [`SingleLinkageTree`](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.SingleLinkageTree "hdbscan.plots.SingleLinkageTree").

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id27 "Link to this heading")

condensed_tree_arraynumpy recarray from `HDBSCAN`

The raw numpy rec array version of the condensed tree as produced internally by hdbscan.

cluster_selection_methodstring, optional (default ‘eom’)

The method of selecting clusters. One of ‘eom’ or ‘leaf’

allow_single_clusterBoolean, optional (default False)

Whether to allow the root cluster as the only selected cluster

get_plot_data(_leaf_separation=1_, _log_size=False_, _max_rectangle_per_icicle=20_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#CondensedTree.get_plot_data)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.CondensedTree.get_plot_data "Link to this definition")

Generates data for use in plotting the ‘icicle plot’ or dendrogram plot of the condensed tree generated by HDBSCAN.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id28 "Link to this heading")

leaf_separationfloat, optional

How far apart to space the final leaves of the dendrogram. (default 1)

log_sizeboolean, optional

Use log scale for the ‘size’ of clusters (i.e. number of points in the cluster at a given lambda value). (default False)

max_rectangles_per_icicleint, optional

To simplify the plot this method will only emit `max_rectangles_per_icicle` bars per branch of the dendrogram. This ensures that we don’t suffer from massive overplotting in cases with a lot of data points.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id29 "Link to this heading")

plot_datadict

Data associated to bars in a bar plot:

bar_centers x coordinate centers for bars bar_tops heights of bars in lambda scalebar_bottoms y coordinate of bottoms of bars bar_widths widths of the bars (in x coord scale) bar_bounds a 4-tuple of [left, right, bottom, top]

> giving the bounds on a full set of cluster bars

Data associates with cluster splits:

line_xs x coordinates for horizontal dendrogram lines line_ys y coordinates for horizontal dendrogram lines

plot(_leaf_separation=1_, _cmap='viridis'_, _select_clusters=False_, _label_clusters=False_, _selection_palette=None_, _axis=None_, _colorbar=True_, _log_size=False_, _max_rectangles_per_icicle=20_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#CondensedTree.plot)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.CondensedTree.plot "Link to this definition")

Use matplotlib to plot an ‘icicle plot’ dendrogram of the condensed tree.

Effectively this is a dendrogram where the width of each cluster bar is equal to the number of points (or log of the number of points) in the cluster at the given lambda value. Thus bars narrow as points progressively drop out of clusters. The make the effect more apparent the bars are also colored according the the number of points (or log of the number of points).

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id30 "Link to this heading")

leaf_separationfloat, optional (default 1)

How far apart to space the final leaves of the dendrogram.

cmapstring or matplotlib colormap, optional (default viridis)

The matplotlib colormap to use to color the cluster bars.

select_clustersboolean, optional (default False)

Whether to draw ovals highlighting which cluster bar represent the clusters that were selected by HDBSCAN as the final clusters.

label_clustersboolean, optional (default False)

If select_clusters is True then this determines whether to draw text labels on the clusters.

selection_palettelist of colors, optional (default None)

If not None, and at least as long as the number of clusters, draw ovals in colors iterating through this palette. This can aid in cluster identification when plotting.

axismatplotlib axis or None, optional (default None)

The matplotlib axis to render to. If None then a new axis will be generated. The rendered axis will be returned.

colorbarboolean, optional (default True)

Whether to draw a matplotlib colorbar displaying the range of cluster sizes as per the colormap.

log_sizeboolean, optional (default False)

Use log scale for the ‘size’ of clusters (i.e. number of points in the cluster at a given lambda value).

max_rectangles_per_icicleint, optional (default 20)

> To simplify the plot this method will only emit `max_rectangles_per_icicle` bars per branch of the dendrogram. This ensures that we don’t suffer from massive overplotting in cases with a lot of data points.

Returns

to_networkx()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#CondensedTree.to_networkx)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.CondensedTree.to_networkx "Link to this definition")

Return a NetworkX DiGraph object representing the condensed tree.

Edge weights in the graph are the lamba values at which child nodes ‘leave’ the parent cluster.

Nodes have a size attribute attached giving the number of points that are in the cluster (or 1 if it is a singleton point) at the point of cluster creation (fewer points may be in the cluster at larger lambda values).

to_numpy()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#CondensedTree.to_numpy)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.CondensedTree.to_numpy "Link to this definition")

Return a numpy structured array representation of the condensed tree.

to_pandas()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#CondensedTree.to_pandas)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.CondensedTree.to_pandas "Link to this definition")

Return a pandas dataframe representation of the condensed tree.

Each row of the dataframe corresponds to an edge in the tree. The columns of the dataframe are parent, child, lambda_val and child_size.

The parent and child are the ids of the parent and child nodes in the tree. Node ids less than the number of points in the original dataset represent individual points, while ids greater than the number of points are clusters.

The lambda_val value is the value (1/distance) at which the child node leaves the cluster.

The child_size is the number of points in the child node.

_class_hdbscan.plots.SingleLinkageTree(_linkage_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#SingleLinkageTree)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.SingleLinkageTree "Link to this definition")

A single linkage format dendrogram tree, with plotting functionality and networkX support.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id31 "Link to this heading")

linkagendarray (n_samples, 4)

The numpy array that holds the tree structure. As output by scipy.cluster.hierarchy, hdbscan, of fastcluster.

get_clusters(_cut_distance_, _min_cluster_size=5_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#SingleLinkageTree.get_clusters)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.SingleLinkageTree.get_clusters "Link to this definition")

Return a flat clustering from the single linkage hierarchy.

This represents the result of selecting a cut value for robust single linkage clustering. The min_cluster_size allows the flat clustering to declare noise points (and cluster smaller than min_cluster_size).

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id32 "Link to this heading")

cut_distancefloat

The mutual reachability distance cut value to use to generate a flat clustering.

min_cluster_sizeint, optional

Clusters smaller than this value with be called ‘noise’ and remain unclustered in the resulting flat clustering.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id33 "Link to this heading")

labelsarray [n_samples]

An array of cluster labels, one per datapoint. Unclustered points are assigned the label -1.

plot(_axis=None_, _truncate_mode=None_, _p=0_, _vary_line_width=True_, _cmap='viridis'_, _colorbar=True_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#SingleLinkageTree.plot)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.SingleLinkageTree.plot "Link to this definition")

Plot a dendrogram of the single linkage tree.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id34 "Link to this heading")

truncate_modestr, optional

The dendrogram can be hard to read when the original observation matrix from which the linkage is derived is large. Truncation is used to condense the dendrogram. There are several modes:

`None/'none'`

No truncation is performed (Default).

`'lastp'`

The last p non-singleton formed in the linkage are the only non-leaf nodes in the linkage; they correspond to rows Z[n-p-2:end] in Z. All other non-singleton clusters are contracted into leaf nodes.

`'level'/'mtica'`

No more than p levels of the dendrogram tree are displayed. This corresponds to Mathematica(TM) behavior.

pint, optional

The `p` parameter for `truncate_mode`.

vary_line_widthboolean, optional

Draw downward branches of the dendrogram with line thickness that varies depending on the size of the cluster.

cmapstring or matplotlib colormap, optional

The matplotlib colormap to use to color the cluster bars. A value of ‘none’ will result in black bars. (default ‘viridis’)

colorbarboolean, optional

Whether to draw a matplotlib colorbar displaying the range of cluster sizes as per the colormap. (default True)

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id35 "Link to this heading")

axismatplotlib axis

The axis on which the dendrogram plot has been rendered.

to_networkx()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#SingleLinkageTree.to_networkx)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.SingleLinkageTree.to_networkx "Link to this definition")

Return a NetworkX DiGraph object representing the single linkage tree.

Edge weights in the graph are the distance values at which child nodes merge to form the parent cluster.

Nodes have a size attribute attached giving the number of points that are in the cluster.

to_numpy()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#SingleLinkageTree.to_numpy)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.SingleLinkageTree.to_numpy "Link to this definition")

Return a numpy array representation of the single linkage tree.

This representation conforms to the scipy.cluster.hierarchy notion of a single linkage tree, and can be used with all the associated scipy tools. Please see the scipy documentation for more details on the format.

to_pandas()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#SingleLinkageTree.to_pandas)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.SingleLinkageTree.to_pandas "Link to this definition")

Return a pandas dataframe representation of the single linkage tree.

Each row of the dataframe corresponds to an edge in the tree. The columns of the dataframe are parent, left_child, right_child, distance and size.

The parent, left_child and right_child are the ids of the parent and child nodes in the tree. Node ids less than the number of points in the original dataset represent individual points, while ids greater than the number of points are clusters.

The distance value is the at which the child nodes merge to form the parent node.

The size is the number of points in the parent node.

_class_hdbscan.plots.MinimumSpanningTree(_mst_, _data_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#MinimumSpanningTree)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.MinimumSpanningTree "Link to this definition")

plot(_axis=None_, _node_size=40_, _node_color='k'_, _node_alpha=0.8_, _edge_alpha=0.5_, _edge_cmap='viridis_r'_, _edge_linewidth=2_, _vary_line_width=True_, _colorbar=True_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#MinimumSpanningTree.plot)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.MinimumSpanningTree.plot "Link to this definition")

Plot the minimum spanning tree (as projected into 2D by t-SNE if required).

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id36 "Link to this heading")

axismatplotlib axis, optional

The axis to render the plot to

node_sizeint, optional

The size of nodes in the plot (default 40).

node_colormatplotlib color spec, optional

The color to render nodes (default black).

node_alphafloat, optional

The alpha value (between 0 and 1) to render nodes with (default 0.8).

edge_cmapmatplotlib colormap, optional

The colormap to color edges by (varying color by edge

weight/distance). Can be a cmap object or a string recognised by matplotlib. (default viridis_r)

edge_alphafloat, optional

The alpha value (between 0 and 1) to render edges with (default 0.5).

edge_linewidthfloat, optional

The linewidth to use for rendering edges (default 2).

vary_line_widthbool, optional

Edge width is proportional to (log of) the inverse of the mutual reachability distance. (default True)

colorbarbool, optional

Whether to draw a colorbar. (default True)

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id37 "Link to this heading")

axismatplotlib axis

The axis used the render the plot.

to_networkx()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#MinimumSpanningTree.to_networkx)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.MinimumSpanningTree.to_networkx "Link to this definition")

Return a NetworkX Graph object representing the minimum spanning tree.

Edge weights in the graph are the distance between the nodes they connect.

Nodes have a data attribute attached giving the data vector of the associated point.

to_numpy()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#MinimumSpanningTree.to_numpy)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.MinimumSpanningTree.to_numpy "Link to this definition")

Return a numpy array of weighted edges in the minimum spanning tree

to_pandas()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#MinimumSpanningTree.to_pandas)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.MinimumSpanningTree.to_pandas "Link to this definition")

Return a Pandas dataframe of the minimum spanning tree.

Each row is an edge in the tree; the columns are from, to, and distance giving the two vertices of the edge which are indices into the dataset, and the distance between those datapoints.

hdbscan.validity.all_points_core_distance(_distance_matrix_, _d=2.0_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/validity.html#all_points_core_distance)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.validity.all_points_core_distance "Link to this definition")

Compute the all-points-core-distance for all the points of a cluster.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id38 "Link to this heading")

distance_matrixarray (cluster_size, cluster_size)

The pairwise distance matrix between points in the cluster.

dinteger

The dimension of the data set, which is used in the computation of the all-point-core-distance as per the paper.

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id39 "Link to this heading")

core_distancesarray (cluster_size,)

The all-points-core-distance of each point in the cluster

### References[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id40 "Link to this heading")

Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J., 2014. Density-Based Clustering Validation. In SDM (pp. 839-847).

hdbscan.validity.density_separation(_X_, _labels_, _cluster_id1_, _cluster_id2_, _internal_nodes1_, _internal_nodes2_, _core_distances1_, _core_distances2_, _metric='euclidean'_, _no_coredist=False_, _**kwd_args_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/validity.html#density_separation)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.validity.density_separation "Link to this definition")

Compute the density separation between two clusters. This is the minimum distance between pairs of points, one from internal nodes of MSTs of each cluster.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id41 "Link to this heading")

Xarray (n_samples, n_features) or (n_samples, n_samples)

The input data of the clustering. This can be the data, or, if metric is set to precomputed the pairwise distance matrix used for the clustering.

labelsarray (n_samples)

The label array output by the clustering, providing an integral cluster label to each data point, with -1 for noise points.

cluster_id1integer

The first cluster label to compute separation between.

cluster_id2integer

The second cluster label to compute separation between.

internal_nodes1array

The vertices of the MST for cluster_id1 that were internal vertices.

internal_nodes2array

The vertices of the MST for cluster_id2 that were internal vertices.

core_distances1array (size of cluster_id1,)

The all-points-core_distances of all points in the cluster specified by cluster_id1.

core_distances2array (size of cluster_id2,)

The all-points-core_distances of all points in the cluster specified by cluster_id2.

metricstring

The metric used to compute distances for the clustering (and to be re-used in computing distances for mr distance). If set to precomputed then X is assumed to be the precomputed distance matrix between samples.

[**](https://hdbscan.readthedocs.io/en/latest/api.html#id42)kwd_args :

Extra arguments to pass to the distance computation for other metrics, such as minkowski, Mahanalobis etc.

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id44 "Link to this heading")

The ‘density separation’ between the clusters specified by cluster_id1 and cluster_id2.

### References[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id45 "Link to this heading")

Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J., 2014. Density-Based Clustering Validation. In SDM (pp. 839-847).

hdbscan.validity.distances_between_points(_X_, _labels_, _cluster_id_, _metric='euclidean'_, _d=None_, _no_coredist=False_, _print_max_raw_to_coredist_ratio=False_, _**kwd_args_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/validity.html#distances_between_points)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.validity.distances_between_points "Link to this definition")

Compute pairwise distances for all the points of a cluster.

If metric is ‘precomputed’ then assume X is a distance matrix for the full dataset. Note that in this case you must pass in ‘d’ the dimension of the dataset.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id46 "Link to this heading")

Xarray (n_samples, n_features) or (n_samples, n_samples)

The input data of the clustering. This can be the data, or, if metric is set to precomputed the pairwise distance matrix used for the clustering.

labelsarray (n_samples)

The label array output by the clustering, providing an integral cluster label to each data point, with -1 for noise points.

cluster_idinteger

The cluster label for which to compute the distances

metricstring

The metric used to compute distances for the clustering (and to be re-used in computing distances for mr distance). If set to precomputed then X is assumed to be the precomputed distance matrix between samples.

dinteger (or None)

The number of features (dimension) of the dataset. This need only be set in the case of metric being set to precomputed, where the ambient dimension of the data is unknown to the function.

[**](https://hdbscan.readthedocs.io/en/latest/api.html#id47)kwd_args :

Extra arguments to pass to the distance computation for other metrics, such as minkowski, Mahanalobis etc.

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id49 "Link to this heading")

distancesarray (n_samples, n_samples)

The distances between all points in X with label equal to cluster_id.

core_distancesarray (n_samples,)

The all-points-core_distance of all points in X with label equal to cluster_id.

### References[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id50 "Link to this heading")

Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J., 2014. Density-Based Clustering Validation. In SDM (pp. 839-847).

hdbscan.validity.internal_minimum_spanning_tree(_mr_distances_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/validity.html#internal_minimum_spanning_tree)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.validity.internal_minimum_spanning_tree "Link to this definition")

Compute the ‘internal’ minimum spanning tree given a matrix of mutual reachability distances. Given a minimum spanning tree the ‘internal’ graph is the subgraph induced by vertices of degree greater than one.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id51 "Link to this heading")

mr_distancesarray (cluster_size, cluster_size)

The pairwise mutual reachability distances, inferred to be the edge weights of a complete graph. Since MSTs are computed per cluster this is the all-points-mutual-reacability for points within a single cluster.

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id52 "Link to this heading")

internal_nodesarray

An array listing the indices of the internal nodes of the MST

internal_edgesarray (?, 3)

An array of internal edges in weighted edge list format; that is an edge is an array of length three listing the two vertices forming the edge and weight of the edge.

### References[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id53 "Link to this heading")

Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J., 2014. Density-Based Clustering Validation. In SDM (pp. 839-847).

hdbscan.validity.validity_index(_X_, _labels_, _metric='euclidean'_, _d=None_, _per_cluster_scores=False_, _mst_raw_dist=False_, _verbose=False_, _**kwd_args_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/validity.html#validity_index)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.validity.validity_index "Link to this definition")

Compute the density based cluster validity index for the clustering specified by labels and for each cluster in labels.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id54 "Link to this heading")

Xarray (n_samples, n_features) or (n_samples, n_samples)

The input data of the clustering. This can be the data, or, if metric is set to precomputed the pairwise distance matrix used for the clustering.

labelsarray (n_samples)

The label array output by the clustering, providing an integral cluster label to each data point, with -1 for noise points.

metricoptional, string (default ‘euclidean’)

The metric used to compute distances for the clustering (and to be re-used in computing distances for mr distance). If set to precomputed then X is assumed to be the precomputed distance matrix between samples.

doptional, integer (or None) (default None)

The number of features (dimension) of the dataset. This need only be set in the case of metric being set to precomputed, where the ambient dimension of the data is unknown to the function.

per_cluster_scoresoptional, boolean (default False)

Whether to return the validity index for individual clusters. Defaults to False with the function returning a single float value for the whole clustering.

mst_raw_distoptional, boolean (default False)

If True, the MST’s are constructed solely via ‘raw’ distances (depending on the given metric, e.g. euclidean distances) instead of using mutual reachability distances. Thus setting this parameter to True avoids using ‘all-points-core-distances’ at all. This is advantageous specifically in the case of elongated clusters that lie in close proximity to each other <citation needed>.

[**](https://hdbscan.readthedocs.io/en/latest/api.html#id55)kwd_args :

Extra arguments to pass to the distance computation for other metrics, such as minkowski, Mahanalobis etc.

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id57 "Link to this heading")

validity_indexfloat

The density based cluster validity index for the clustering. This is a numeric value between -1 and 1, with higher values indicating a ‘better’ clustering.

per_cluster_validity_indexarray (n_clusters,)

The cluster validity index of each individual cluster as an array. The overall validity index is the weighted average of these values. Only returned if per_cluster_scores is set to True.

### References[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id58 "Link to this heading")

Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J., 2014. Density-Based Clustering Validation. In SDM (pp. 839-847).

_class_hdbscan.prediction.PredictionData(_data_, _condensed_tree_, _min_samples_, _tree_type='kdtree'_, _metric='euclidean'_, _**kwargs_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/prediction.html#PredictionData)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.prediction.PredictionData "Link to this definition")

Extra data that allows for faster prediction if cached.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id59 "Link to this heading")

dataarray (n_samples, n_features)

The original data set that was clustered

condensed_treeCondensedTree

The condensed tree object created by a clustering

min_samplesint

The min_samples value used in clustering

tree_typestring, optional

Which type of space tree to use for core distance computation. One of:

> - `kdtree`
>     
> - `balltree`
>     

metricstring, optional

The metric used to determine distance for the clustering. This is the metric that will be used for the space tree to determine core distances etc.

[**](https://hdbscan.readthedocs.io/en/latest/api.html#id60)kwargs :

Any further arguments to the metric.

### Attributes[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id62 "Link to this heading")

raw_dataarray (n_samples, n_features)

The original data set that was clustered

treeKDTree or BallTree

A space partitioning tree that can be queried for nearest neighbors.

core_distancesarray (n_samples,)

The core distances for every point in the original data set.

cluster_mapdict

A dictionary mapping cluster numbers in the condensed tree to labels in the final selected clustering.

cluster_treestructured array

A version of the condensed tree that only contains clusters, not individual points.

max_lambdasdict

A dictionary mapping cluster numbers in the condensed tree to the maximum lambda value seen in that cluster.

hdbscan.prediction.all_points_membership_vectors(_clusterer_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/prediction.html#all_points_membership_vectors)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.prediction.all_points_membership_vectors "Link to this definition")

Predict soft cluster membership vectors for all points in the original dataset the clusterer was trained on. This function is more efficient by making use of the fact that all points are already in the condensed tree, and processing in bulk.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id63 "Link to this heading")

clustererHDBSCAN

> A clustering object that has been fit to the data and

either had `prediction_data=True` set, or called the `generate_prediction_data` method after the fact. This method does not work if the clusterer was trained with `metric='precomputed'`.

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id64 "Link to this heading")

membership_vectorsarray (n_samples, n_clusters)

The probability that point `i` of the original dataset is a member of cluster `j` is in `membership_vectors[i, j]`.

### See Also[¶](https://hdbscan.readthedocs.io/en/latest/api.html#see-also "Link to this heading")

`hdbscan.predict.predict()` `hdbscan.predict.all_points_membership_vectors()`

hdbscan.prediction.approximate_predict(_clusterer_, _points_to_predict_, _return_connecting_points=False_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/prediction.html#approximate_predict)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.prediction.approximate_predict "Link to this definition")

Predict the cluster label of new points. The returned labels will be those of the original clustering found by `clusterer`, and therefore are not (necessarily) the cluster labels that would be found by clustering the original data combined with `points_to_predict`, hence the ‘approximate’ label.

If you simply wish to assign new points to an existing clustering in the ‘best’ way possible, this is the function to use. If you want to predict how `points_to_predict` would cluster with the original data under HDBSCAN the most efficient existing approach is to simply recluster with the new point(s) added to the original dataset.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id65 "Link to this heading")

clustererHDBSCAN

A clustering object that has been fit to the data and either had `prediction_data=True` set, or called the `generate_prediction_data` method after the fact.

points_to_predictarray, or array-like (n_samples, n_features)

The new data points to predict cluster labels for. They should have the same dimensionality as the original dataset over which clusterer was fit.

return_connecting_pointsbool, optional

Whether to return the index of the nearest neighbor in the original dataset for each of the `points_to_predict`. Default is False

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id66 "Link to this heading")

labelsarray (n_samples,)

The predicted labels of the `points_to_predict`

probabilitiesarray (n_samples,)

The soft cluster scores for each of the `points_to_predict`

neighborsarray (n_samples,)

The index of the nearest neighbor in the original dataset for each of the `points_to_predict`. Only returned if `return_connecting_points=True`.

### See Also[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id67 "Link to this heading")

`hdbscan.predict.membership_vector()` `hdbscan.predict.all_points_membership_vectors()`

hdbscan.prediction.approximate_predict_scores(_clusterer_, _points_to_predict_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/prediction.html#approximate_predict_scores)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.prediction.approximate_predict_scores "Link to this definition")

Predict the outlier score of new points. The returned scores will be based on the original clustering found by `clusterer`, and therefore are not (necessarily) the outlier scores that would be found by clustering the original data combined with `points_to_predict`, hence the ‘approximate’ label.

If you simply wish to calculate the outlier scores for new points in the ‘best’ way possible, this is the function to use. If you want to predict the outlier score of `points_to_predict` with the original data under HDBSCAN the most efficient existing approach is to simply recluster with the new point(s) added to the original dataset.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id68 "Link to this heading")

clustererHDBSCAN

A clustering object that has been fit to the data and either had `prediction_data=True` set, or called the `generate_prediction_data` method after the fact.

points_to_predictarray, or array-like (n_samples, n_features)

The new data points to predict cluster labels for. They should have the same dimensionality as the original dataset over which clusterer was fit.

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id69 "Link to this heading")

scoresarray (n_samples,)

The predicted scores of the `points_to_predict`

### See Also[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id70 "Link to this heading")

`hdbscan.predict.membership_vector()` `hdbscan.predict.all_points_membership_vectors()`

hdbscan.prediction.membership_vector(_clusterer_, _points_to_predict_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/prediction.html#membership_vector)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.prediction.membership_vector "Link to this definition")

Predict soft cluster membership. The result produces a vector for each point in `points_to_predict` that gives a probability that the given point is a member of a cluster for each of the selected clusters of the `clusterer`.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id71 "Link to this heading")

clustererHDBSCAN

A clustering object that has been fit to the data and either had `prediction_data=True` set, or called the `generate_prediction_data` method after the fact.

points_to_predictarray, or array-like (n_samples, n_features)

The new data points to predict cluster labels for. They should have the same dimensionality as the original dataset over which clusterer was fit.

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id72 "Link to this heading")

membership_vectorsarray (n_samples, n_clusters)

The probability that point `i` is a member of cluster `j` is in `membership_vectors[i, j]`.

### See Also[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id73 "Link to this heading")

`hdbscan.predict.predict()` `hdbscan.predict.all_points_membership_vectors()`

## Branch detection[¶](https://hdbscan.readthedocs.io/en/latest/api.html#branch-detection "Link to this heading")

The branches module contains classes for detecting branches within clusters.

_class_hdbscan.branches.BranchDetector(_branch_detection_method='full'_, _label_sides_as_branches=False_, _min_cluster_size=None_, _max_cluster_size=None_, _allow_single_cluster=None_, _cluster_selection_method=None_, _cluster_selection_epsilon=0.0_, _cluster_selection_persistence=0.0_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/branches.html#BranchDetector)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector "Link to this definition")

Performs a flare-detection post-processing step to detect branches within clusters [[1]_](https://hdbscan.readthedocs.io/en/latest/api.html#id136).

For each cluster, a graph is constructed connecting the data points based on their mutual reachability distances. Each edge is given a centrality value based on how far it lies from the cluster’s center. Then, the edges are clustered as if that centrality was a distance, progressively removing the ‘center’ of each cluster and seeing how many branches remain.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id75 "Link to this heading")

branch_detection_methodstr, optional (default=``full``)

Determines which graph is constructed to detect branches with. Valid values are, ordered by increasing computation cost and decreasing sensitivity to noise: - `core`: Contains the edges that connect each point to all other

> points within a mutual reachability distance lower than or equal to the point’s core distance. This is the cluster’s subgraph of the k-NN graph over the entire data set (with k = `min_samples`).

- `full`: Contains all edges between points in each cluster with a mutual reachability distance lower than or equal to the distance of the most-distance point in each cluster. These graphs represent the 0-dimensional simplicial complex of each cluster at the first point in the filtration where they contain all their points.
    

label_sides_as_branchesbool, optional (default=False),

When this flag is False, branches are only labelled for clusters with at least three branches (i.e., at least y-shapes). Clusters with only two branches represent l-shapes. The two branches describe the cluster’s outsides growing towards each other. Enabling this flag separates these branches from each other in the produced labelling.

min_cluster_sizeint, optional (default=None)

The minimum number of samples in a group for that group to be considered a branch; groupings smaller than this size will seen as points falling out of a branch. Defaults to the clusterer’s min_cluster_size.

allow_single_clusterbool, optional (default=None)

Analogous to `allow_single_cluster`.

cluster_selection_methodstr, optional (default=None)

The method used to select branches from the cluster’s condensed tree. The standard approach for FLASC is to use the `eom` approach. Options are:

> - `eom`
>     
> - `leaf`
>     

cluster_selection_epsilon: float, optional (default=0.0)

A lower epsilon threshold. Only branches with a death above this value will be considered.

cluster_selection_persistence: float, optional (default=0.0)

An eccentricity persistence threshold. Branches with a persistence below this value will be merged.

max_cluster_sizeint, optional (default=None)

A limit to the size of clusters returned by the `eom` algorithm. Has no effect when using `leaf` clustering (where clusters are usually small regardless). Note that this should not be used if we want to predict the cluster labels for new points in future becauseapproximate_predict is not aware of this argument.

### Attributes[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id76 "Link to this heading")

[labels_](https://hdbscan.readthedocs.io/en/latest/api.html#id137)np.ndarray, shape (n_samples, )

Labels that differentiate all subgroups (clusters and branches). Noisy samples are given the label -1.

[probabilities_](https://hdbscan.readthedocs.io/en/latest/api.html#id139)np.ndarray, shape (n_samples, )

Probabilities considering both cluster and branch membership. Noisy samples are assigned 0.

[cluster_labels_](https://hdbscan.readthedocs.io/en/latest/api.html#id141)np.ndarray, shape (n_samples, )

The cluster labels for each point in the data set. Noisy samples are given the label -1.

[cluster_probabilities_](https://hdbscan.readthedocs.io/en/latest/api.html#id143)np.ndarray, shape (n_samples, )

The cluster probabilities for each point in the data set. Noisy samples are assigned 1.0.

[branch_labels_](https://hdbscan.readthedocs.io/en/latest/api.html#id145)np.ndarray, shape (n_samples, )

Branch labels for each point. Noisy samples are given the label -1.

[branch_probabilities_](https://hdbscan.readthedocs.io/en/latest/api.html#id147)np.ndarray, shape (n_samples, )

Branch membership strengths for each point. Noisy samples are assigned 0.

[branch_persistences_](https://hdbscan.readthedocs.io/en/latest/api.html#id149)tuple (n_clusters)

A branch persistence (eccentricity range) for each detected branch.

[approximation_graph_](https://hdbscan.readthedocs.io/en/latest/api.html#id151)ApproximationGraph

The graphs used to detect branches in each cluster stored as a numpy array with four columns: source, target, centrality, mutual reachability distance. Points are labelled by their row-index into the input data. The edges contained in the graphs depend on the `branch_detection_method`: - `core`: Contains the edges that connect each point to all other

> points in a cluster within a mutual reachability distance lower than or equal to the point’s core distance. This is an extension of the minimum spanning tree introducing only edges with equal distances. The reachability distance introduces `num_points` * `min_samples` of such edges.

- `full`: Contains all edges between points in each cluster with a mutual reachability distance lower than or equal to the distance of the most-distance point in each cluster. These graphs represent the 0-dimensional simplicial complex of each cluster at the first point in the filtration where they contain all their points.
    

[condensed_trees_](https://hdbscan.readthedocs.io/en/latest/api.html#id153)tuple (n_clusters)

A condensed branch hierarchy for each cluster produced during the branch detection step. Data points are numbered with in-cluster ids.

[linkage_trees_](https://hdbscan.readthedocs.io/en/latest/api.html#id155)tuple (n_clusters)

A single linkage tree for each cluster produced during the branch detection step, in the scipy hierarchical clustering format. (see [http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html](http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)). Data points are numbered with in-cluster ids.

[centralities_](https://hdbscan.readthedocs.io/en/latest/api.html#id157)np.ndarray, shape (n_samples, )

Centrality values for each point in a cluster. Overemphasizes points’ eccentricity within the cluster as the values are based on minimum spanning trees that do not contain the equally distanced edges resulting from the mutual reachability distance.

[cluster_points_](https://hdbscan.readthedocs.io/en/latest/api.html#id159)list (n_clusters)

The data point row indices for each cluster.

### References[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id77 "Link to this heading")

[1]

Bot D.M., Peeters J., Liesenborgs J., Aerts J. 2025. FLASC: a

flare-sensitive clustering algorithm. PeerJ Computer Science 11:e2792[https://doi.org/10.7717/peerj-cs.2792](https://doi.org/10.7717/peerj-cs.2792).

_property_approximation_graph_[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector.approximation_graph_ "Link to this definition")

See [`BranchDetector`](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector "hdbscan.branches.BranchDetector") for documentation.

 

_property_condensed_trees_[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector.condensed_trees_ "Link to this definition")

See [`BranchDetector`](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector "hdbscan.branches.BranchDetector") for documentation.

_property_exemplars_[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector.exemplars_ "Link to this definition")

See [`BranchDetector`](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector "hdbscan.branches.BranchDetector") for documentation.

fit(_clusterer_, _labels=None_, _probabilities=None_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/branches.html#BranchDetector.fit)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector.fit "Link to this definition")

Perform a flare-detection post-processing step to detect branches within clusters.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id79 "Link to this heading")

clustererHDBSCAN

A fitted HDBSCAN object with branch detection data generated.

labelsnp.ndarray, shape (n_samples, ), optional (default=None)

The cluster labels for each point in the data set. If not provided, the clusterer’s labels will be used.

probabilitiesnp.ndarray, shape (n_samples, ), optional (default=None)

The cluster probabilities for each point in the data set. If not provided, the clusterer’s probabilities will be used, or all points will be given 1.0 probability if labels are overridden.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id80 "Link to this heading")

selfobject

Returns self.

fit_predict(_clusterer_, _labels=None_, _probabilities=None_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/branches.html#BranchDetector.fit_predict)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector.fit_predict "Link to this definition")

Perform a flare-detection post-processing step to detect branches within clusters [[1]_](https://hdbscan.readthedocs.io/en/latest/api.html#id161).

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id82 "Link to this heading")

clustererHDBSCAN

A fitted HDBSCAN object with branch detection data generated.

labelsnp.ndarray, shape (n_samples, ), optional (default=None)

The cluster labels for each point in the data set. If not provided, the clusterer’s labels will be used.

probabilitiesnp.ndarray, shape (n_samples, ), optional (default=None)

The cluster probabilities for each point in the data set. If not provided, the clusterer’s probabilities will be used, or all points will be given 1.0 probability if labels are overridden.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id83 "Link to this heading")

labelsndarray, shape (n_samples, )

subgroup labels differentiated by cluster and branch.

_property_linkage_trees_[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector.linkage_trees_ "Link to this definition")

See [`BranchDetector`](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector "hdbscan.branches.BranchDetector") for documentation.

set_fit_request(_*_, _clusterer: bool | None | str = '$UNCHANGED$'_, _labels: bool | None | str ='$UNCHANGED$'_, _probabilities: bool | None | str = '$UNCHANGED$'_)→ [BranchDetector](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector "hdbscan.branches.BranchDetector")[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector.set_fit_request "Link to this definition")

Request metadata passed to the `fit` method.

Note that this method is only relevant if `enable_metadata_routing=True` (see `sklearn.set_config()`). Please see User Guide on how the routing mechanism works.

The options for each parameter are:

- `True`: metadata is requested, and passed to `fit` if provided. The request is ignored if metadata is not provided.
    
- `False`: metadata is not requested and the meta-estimator will not pass it to `fit`.
    
- `None`: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
    
- `str`: metadata should be passed to the meta-estimator with this given alias instead of the original name.
    

The default (`sklearn.utils.metadata_routing.UNCHANGED`) retains the existing request. This allows you to change the request for some parameters and not others.

Added in version 1.3.

Note

This method is only relevant if this estimator is used as a sub-estimator of a meta-estimator, e.g. used inside a `Pipeline`. Otherwise it has no effect.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id84 "Link to this heading")

clustererstr, True, False, or None, default=sklearn.utils.metadata_routing.UNCHANGED

Metadata routing for `clusterer` parameter in `fit`.

labelsstr, True, False, or None, default=sklearn.utils.metadata_routing.UNCHANGED

Metadata routing for `labels` parameter in `fit`.

probabilitiesstr, True, False, or None, default=sklearn.utils.metadata_routing.UNCHANGED

Metadata routing for `probabilities` parameter in `fit`.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id85 "Link to this heading")

selfobject

The updated object.

weighted_centroid(_label_id_, _data=None_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/branches.html#BranchDetector.weighted_centroid)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector.weighted_centroid "Link to this definition")

Provides an approximate representative point for a given branch. Note that this technique assumes a euclidean metric for speed of computation. For more general metrics use the `weighted_medoid` method which is slower, but can work with the metric the model trained with.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id86 "Link to this heading")

label_id: int

The id of the cluster to compute a centroid for.

datanp.ndarray (n_samples, n_features), optional (default=None)

A dataset to use instead of the raw data that was clustered on.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id87 "Link to this heading")

centroid: array of shape (n_features,)

A representative centroid for cluster `label_id`.

weighted_medoid(_label_id_, _data=None_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/branches.html#BranchDetector.weighted_medoid)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.BranchDetector.weighted_medoid "Link to this definition")

Provides an approximate representative point for a given branch.

Note that this technique can be very slow and memory intensive for large clusters. For faster results use the `weighted_centroid` method which is faster, but assumes a euclidean metric.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id88 "Link to this heading")

label_id: int

The id of the cluster to compute a medoid for.

datanp.ndarray (n_samples, n_features), optional (default=None)

A dataset to use instead of the raw data that was clustered on.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id89 "Link to this heading")

centroid: array of shape (n_features,)

A representative medoid for cluster `label_id`.

hdbscan.branches.approximate_predict_branch(_branch_detector_, _points_to_predict_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/branches.html#approximate_predict_branch)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.approximate_predict_branch "Link to this definition")

Predict the cluster and branch label of new points.

Extends `approximate_predict` to also predict in which branch new points lie (if the cluster they are part of has branches).

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id90 "Link to this heading")

branch_detectorBranchDetector

A clustering object that has been fit to vector input data.

points_to_predictarray, or array-like (n_samples, n_features)

The new data points to predict cluster labels for. They should have the same dimensionality as the original dataset over which clusterer was fit.

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id91 "Link to this heading")

labelsarray (n_samples,)

The predicted cluster and branch labels.

probabilitiesarray (n_samples,)

The soft cluster scores for each.

cluster_labelsarray (n_samples,)

The predicted cluster labels.

cluster_probabilitiesarray (n_samples,)

The soft cluster scores for each.

branch_labelsarray (n_samples,)

The predicted cluster labels.

branch_probabilitiesarray (n_samples,)

The soft cluster scores for each.

hdbscan.branches.detect_branches_in_clusters(_clusterer_, _cluster_labels=None_, _cluster_probabilities=None_, _branch_detection_method='full'_, _label_sides_as_branches=False_, _min_cluster_size=None_, _max_cluster_size=None_, _allow_single_cluster=None_, _cluster_selection_method=None_, _cluster_selection_epsilon=0.0_, _cluster_selection_persistence=0.0_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/branches.html#detect_branches_in_clusters)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.branches.detect_branches_in_clusters "Link to this definition")

Performs a flare-detection post-processing step to detect branches within clusters [[1]_](https://hdbscan.readthedocs.io/en/latest/api.html#id162).

For each cluster, a graph is constructed connecting the data points based on their mutual reachability distances. Each edge is given a centrality value based on how far it lies from the cluster’s center. Then, the edges are clustered as if that centrality was a distance, progressively removing the ‘center’ of each cluster and seeing how many branches remain.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id93 "Link to this heading")

clustererhdbscan.HDBSCAN

The clusterer object that has been fit to the data with branch detection data generated.

cluster_labelsnp.ndarray, shape (n_samples, ), optional (default=None)

The cluster labels for each point in the data set. If not provided, the clusterer’s labels will be used.

cluster_probabilitiesnp.ndarray, shape (n_samples, ), optional (default=None)

The cluster probabilities for each point in the data set. If not provided, the clusterer’s probabilities will be used, or all points will be given 1.0 probability if labels are overridden.

branch_detection_methodstr, optional (default=``full``)

Determines which graph is constructed to detect branches with. Valid values are, ordered by increasing computation cost and decreasing sensitivity to noise: - `core`: Contains the edges that connect each point to all other

> points within a mutual reachability distance lower than or equal to the point’s core distance. This is the cluster’s subgraph of the k-NN graph over the entire data set (with k = `min_samples`).

- `full`: Contains all edges between points in each cluster with a mutual reachability distance lower than or equal to the distance of the most-distance point in each cluster. These graphs represent the 0-dimensional simplicial complex of each cluster at the first point in the filtration where they contain all their points.
    

label_sides_as_branchesbool, optional (default=False),

When this flag is False, branches are only labelled for clusters with at least three branches (i.e., at least y-shapes). Clusters with only two branches represent l-shapes. The two branches describe the cluster’s outsides growing towards each other. Enabling this flag separates these branches from each other in the produced labelling.

min_cluster_sizeint, optional (default=None)

The minimum number of samples in a group for that group to be considered a branch; groupings smaller than this size will seen as points falling out of a branch. Defaults to the clusterer’s min_cluster_size.

allow_single_clusterbool, optional (default=None)

Analogous to HDBSCAN’s `allow_single_cluster`.

cluster_selection_methodstr, optional (default=None)

The method used to select branches from the cluster’s condensed tree. The standard approach for FLASC is to use the `eom` approach. Options are:

> - `eom`
>     
> - `leaf`
>     

cluster_selection_epsilon: float, optional (default=0.0)

A lower epsilon threshold. Only branches with a death above this value will be considered. See [3](https://hdbscan.readthedocs.io/en/latest/api.html#id7) for more information. Note that this should not be used if we want to predict the cluster labels for new points in future (e.g. using approximate_predict), as the`approximate_predict()` function is not aware of this argument.

cluster_selection_persistence: float, optional (default=0.0)

An eccentricity persistence threshold. Branches with a persistence below this value will be merged. See [3](https://hdbscan.readthedocs.io/en/latest/api.html#id7) for more information. Note that this should not be used if we want to predict the cluster labels for new points in future (e.g. using approximate_predict), as the`approximate_predict()` function is not aware of this argument.

max_cluster_sizeint, optional (default=0)

A limit to the size of clusters returned by the `eom` algorithm. Has no effect when using `leaf` clustering (where clusters are usually small regardless). Note that this should not be used if we want to predict the cluster labels for new points in future (e.g. using`approximate_predict()`), as that function is not aware of this argument.

### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id96 "Link to this heading")

labelsnp.ndarray, shape (n_samples, )

Labels that differentiate all subgroups (clusters and branches). Noisy samples are given the label -1.

probabilitiesnp.ndarray, shape (n_samples, )

Probabilities considering both cluster and branch membership. Noisy samples are assigned 0.

cluster_labelsnp.ndarray, shape (n_samples, )

The cluster labels for each point in the data set. Noisy samples are given the label -1.

cluster_probabilitiesnp.ndarray, shape (n_samples, )

The cluster probabilities for each point in the data set. Noisy samples are assigned 1.0.

branch_labelsnp.ndarray, shape (n_samples, )

Branch labels for each point. Noisy samples are given the label -1.

branch_probabilitiesnp.ndarray, shape (n_samples, )

Branch membership strengths for each point. Noisy samples are assigned 0.

branch_persistencestuple (n_clusters)

A branch persistence (eccentricity range) for each detected branch.

approximation_graphstuple (n_clusters)

The graphs used to detect branches in each cluster stored as a numpy array with four columns: source, target, centrality, mutual reachability distance. Points are labelled by their row-index into the input data. The edges contained in the graphs depend on the `branch_detection_method`: - `core`: Contains the edges that connect each point to all other

> points in a cluster within a mutual reachability distance lower than or equal to the point’s core distance. This is an extension of the minimum spanning tree introducing only edges with equal distances. The reachability distance introduces `num_points` * `min_samples` of such edges.

- `full`: Contains all edges between points in each cluster with a mutual reachability distance lower than or equal to the distance of the most-distance point in each cluster. These graphs represent the 0-dimensional simplicial complex of each cluster at the first point in the filtration where they contain all their points.
    

condensed_treestuple (n_clusters)

A condensed branch hierarchy for each cluster produced during the branch detection step. Data points are numbered with in-cluster ids.

linkage_treestuple (n_clusters)

A single linkage tree for each cluster produced during the branch detection step, in the scipy hierarchical clustering format. (see [http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html](http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)). Data points are numbered with in-cluster ids.

centralitiesnp.ndarray, shape (n_samples, )

Centrality values for each point in a cluster. Overemphasizes points’ eccentricity within the cluster as the values are based on minimum spanning trees that do not contain the equally distanced edges resulting from the mutual reachability distance.

cluster_pointslist (n_clusters)

The data point row indices for each cluster.

### References[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id97 "Link to this heading")

[1]

Bot D.M., Peeters J., Liesenborgs J., Aerts J. 2025. FLASC: a

flare-sensitive clustering algorithm. PeerJ Computer Science 11:e2792[https://doi.org/10.7717/peerj-cs.2792](https://doi.org/10.7717/peerj-cs.2792).

_class_hdbscan.plots.ApproximationGraph(_approximation_graphs_, _labels_, _probabilities_, _lens_values_, _cluster_labels=None_, _cluster_probabilities=None_, _sub_cluster_labels=None_, _sub_cluster_probabilities=None_, _*_, _lens_name=None_, _sub_cluster_name=None_, _raw_data=None_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#ApproximationGraph)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.ApproximationGraph "Link to this definition")

Cluster approximation graph describing the connectivity in clusters that is used to detect branches.

### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id99 "Link to this heading")

approximation_graphs : list[np.ndarray], shape (n_clusters),

labelsnp.ndarray, shape (n_samples, )

Data point cluster membership labels.

probabilitiesnp.ndarray, shape (n_samples, )

Data point cluster membership strengths.

lens_valuesnp.ndarray, shape (n_samples, )

Data point lens values used to compute (sub-)clusters.

cluster_labelsnp.ndarray, shape (n_samples, ), optional

The cluster labelling used to compute sub-clusters.

cluster_probabilitiesnp.ndarray, shape (n_samples, ), optional

The cluster probabilities used to compute sub-clusters.

sub_cluster_labelsnp.ndarray, shape (n_samples, ), optional

Labels indicating sub-clusters within clusters.

sub_cluster_probabilitiesnp.ndarray, shape (n_samples, ), optional

Sub-cluster probability.

lens_namestr, optional

The name of the lens used to compute the clusters.

sub_cluster_namestr, optional

The name to use for sub-clusters, e.g. “branch”.

### Attributes[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id100 "Link to this heading")

point_masknp.ndarray[bool], shape (n_samples)

A mask to extract points within clusters from the raw data.

plot(_positions=None_, _feature_names=None_, _node_color='label'_, _node_vmin=None_, _node_vmax=None_, _node_cmap='viridis'_, _node_alpha=1_, _node_size=1_, _node_marker='o'_, _edge_color='k'_, _edge_vmin=None_, _edge_vmax=None_, _edge_cmap='viridis'_, _edge_alpha=1_, _edge_width=1_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#ApproximationGraph.plot)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.ApproximationGraph.plot "Link to this definition")

Plots the Approximation graph, requires networkx and matplotlib.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id101 "Link to this heading")

positionsnp.ndarray, shape (n_samples, 2) (default = None)

A position for each data point in the graph or each data point in the raw data. When None, the function attempts to compute graphviz’ sfdp layout, which requires pygraphviz to be installed and available.

node_colorstr (default = ‘label’)

The point attribute to to color the nodes by. Possible values: - id - label - probability - [lens_name] (default = ‘lens_value’) - cluster_label (if available) - cluster_probability (if available) - [sub_cluster_name]_label (if available, default = ‘sub_cluster’) - [sub_cluster_name]_probability (if available, default = ‘sub_cluster’) - The input data’s feature (if available) names if `feature_names` is specified or `feature_x` for the x-th feature if no `feature_names` are given, or anything matplotlib scatter interprets as a color.

node_vminfloat, (default = None)

The minimum value to use for normalizing node colors.

node_vmaxfloat, (default = None)

The maximum value to use for normalizing node colors.

node_cmapstr, (default = ‘tab10’)

The cmap to use for coloring nodes.

node_alphafloat, (default = 1)

The node transparency value.

node_sizefloat, (default = 5)

The node marker size value.

node_markerstr, (default = ‘o’)

The node marker string.

edge_colorstr (default = ‘label’)

The point attribute to to color the nodes by. Possible values: - weight - mutual reachability - centrality, - cluster, or anything matplotlib linecollection interprets as color.

edge_vminfloat, (default = None)

The minimum value to use for normalizing edge colors.

edge_vmaxfloat, (default = None)

The maximum value to use for normalizing edge colors.

edge_cmapstr, (default = viridis)

The cmap to use for coloring edges.

edge_alphafloat, (default = 1)

The edge transparency value.

edge_widthfloat, (default = 1)

The edge line width size value.

to_networkx(_feature_names=None_)[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#ApproximationGraph.to_networkx)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.ApproximationGraph.to_networkx "Link to this definition")

Convert to a NetworkX Graph object.

#### Parameters[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id102 "Link to this heading")

feature_nameslist[n_features]

Names to use for the data features if available.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id103 "Link to this heading")

gnx.Graph

A NetworkX Graph object containing the non-noise points and edges within clusters.

Node attributes: - label, - probability, - cluster label, - cluster probability, - cluster centrality, - branch label, - branch probability,

Edge attributes: - weight (1 / (1 + mutual_reachability)), - mutual_reachability, - centrality, - cluster label, -

to_numpy()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#ApproximationGraph.to_numpy)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.ApproximationGraph.to_numpy "Link to this definition")

Converts the approximation graph to numpy arrays.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id104 "Link to this heading")

pointsnp.recarray, shape (n_points, 8)

A numpy record array with for each point its: - id (row index), - label, - probability, - cluster label, - cluster probability, - cluster centrality, - branch label, - branch probability

edgesnp.recarray, shape (n_edges, 5)

A numpy record array with for each edge its: - parent point, - child point, - cluster centrality, - mutual reachability, - cluster label

to_pandas()[[source]](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/plots.html#ApproximationGraph.to_pandas)[¶](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.plots.ApproximationGraph.to_pandas "Link to this definition")

Converts the approximation graph to pandas data frames.

#### Returns[¶](https://hdbscan.readthedocs.io/en/latest/api.html#id105 "Link to this heading")

pointspd.DataFrame, shape (n_points, 8)

A DataFrame with for each point its: - id (row index), - label, - probability, - cluster label, - cluster probability, - cluster centrality, - branch label, - branch probability

edgespd.DataFrame, shape (n_edges, 5)

A DataFrame with for each edge its: - parent point, - child point, - cluster centrality, - mutual reachability, - cluster label