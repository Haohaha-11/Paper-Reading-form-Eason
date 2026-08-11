# Deep Sets

Manzil Zaheer<sup>1,2</sup>, Satwik Kottur<sup>1</sup>, Siamak Ravanbhakhsh<sup>1</sup>,

Barnabás Póczos<sup>1</sup>, Ruslan Salakhutdinov<sup>1</sup>, Alexander J Smola<sup>1,2</sup>

<sup>1</sup> Carnegie Mellon University <sup>2</sup> Amazon Web Services

{manzilz,skottur,mravanba,bapoczos,rsalakhu,smola}@cs.cmu.edu

## Abstract

We study the problem of designing models for machine learning tasks defined on sets. In contrast to traditional approach of operating on fixed dimensional vectors, we consider objective functions defined on sets that are invariant to permutations. Such problems are widespread, ranging from estimation of population statistics [1], to anomaly detection in piezometer data of embankment dams [2], to cosmology [3, 4]. Our main theorem characterizes the permutation invariant functions and provides a family of functions to which any permutation invariant objective function must belong. This family of functions has a special structure which enables us to design a deep network architecture that can operate on sets and which can be deployed on a variety of scenarios including both unsupervised and supervised learning tasks. We also derive the necessary and sufficient conditions for permutation equivariance in deep models. We demonstrate the applicability of our method on population statistic estimation, point cloud classification, set expansion, and outlier detection.

## 1 Introduction

A typical machine learning algorithm, like regression or classification, is designed for fixed dimensional data instances. Their extensions to handle the case when the inputs or outputs are permutation invariant sets rather than fixed dimensional vectors is not trivial and researchers have only recently started to investigate them [5–8]. In this paper, we present a generic framework to deal with the setting where input and possibly output instances in a machine learning task are sets.

Similar to fixed dimensional data instances, we can characterize two learning paradigms in case of sets. In supervised learning, we have an output label for a set that is invariant or equivariant to the permutation of set elements. Examples include tasks like estimation of population statistics [1], where applications range from giga-scale cosmology [3, 4] to nano-scale quantum chemistry [9].

Next, there can be the unsupervised setting, where the “set” structure needs to be learned, e.g. by leveraging the homophily/heterophily tendencies within sets. An example is the task of set expansion (a.k.a. audience expansion), where given a set of objects that are similar to each other (e.g. set of words {lion, tiger, leopard}), our goal is to find new objects from a large pool of candidates such that the selected new objects are similar to the query set (e.g. find words like jaguar or cheetah among all English words). This is a standard problem in similarity search and metric learning, and a typical application is to find new image tags given a small set of possible tags. Likewise, in the field of computational advertisement, given a set of high-value customers, the goal would be to find similar people. This is an important problem in many scientific applications, e.g. given a small set of interesting celestial objects, astrophysicists might want to find similar ones in large sky surveys.

Main contributions. In this paper, (i) we propose a fundamental architecture, DeepSets, to deal with sets as inputs and show that the properties of this architecture are both necessary and sufficient (Sec. 2). (ii) We extend this architecture to allow for conditioning on arbitrary objects, and (iii) based on this architecture we develop a deep network that can operate on sets with possibly different sizes (Sec. 3). We show that a simple parameter-sharing scheme enables a general treatment of sets within supervised and semi-supervised settings. (iv) Finally, we demonstrate the wide applicability of our framework through experiments on diverse problems (Sec. 4).

## 2 Permutation Invariance and Equivariance

## 2.1 Problem Definition

A function f transforms its domain X into its range Y. Usually, the input domain is a vector space $\mathbb { R } ^ { d }$ and the output response range is either a discrete space, e.g. {0, 1} in case of classification, or a continuous space R in case of regression. Now, if the input is a set $X = \{ x _ { 1 } , \ldots , x _ { M } \} , x _ { m } \in \mathfrak { X }$ i.e., the input domain is the power set $\mathcal { X } = 2 ^ { \mathfrak { X } }$ , then we would like the response of the function to be “indifferent” to the ordering of the elements. In other words,

Property 1 A function $f : 2 ^ { \mathfrak { X } } \to \mathcal { Y }$ acting on sets must be permutation invariant to the order of objects in the set, i.e.for any permutation π $\cdot : f ( \{ x _ { 1 } , \ldots , x _ { M } \} ) = f ( \{ x _ { \pi ( 1 ) } , \ldots , x _ { \pi ( M ) } \} )$ In the supervised setting, given N examples of of $X ^ { ( 1 ) } , . . . , X ^ { ( N ) }$ as well as their labels $y ^ { ( 1 ) } , . . . , y ^ { ( N ) }$ the task would be to classify/regress (with variable number of predictors) while being permutation invariant w.r.t. predictors. Under unsupervised setting, the task would be to assign high scores to valid sets and low scores to improbable sets. These scores can then be used for set expansion tasks, such as image tagging or audience expansion in field of computational advertisement. In transductive setting, each instance $x _ { m } ^ { ( n ) }$ has an associated labeled $y _ { m } ^ { ( n ) }$ . Then, the objective would be instead to learn a permutation equivariant function $\mathbf { f } : \mathfrak { X } ^ { M } \to \mathcal { Y } ^ { M }$ that upon permutation of the input instances permutes the output labels, i.e. for any permutation π:

$$
\mathbf { f } ( [ x _ { \pi ( 1 ) } , \hdots , x _ { \pi ( M ) } ] ) = [ f _ { \pi ( 1 ) } ( \mathbf { x } ) , \hdots , f _ { \pi ( M ) } ( \mathbf { x } ) ]\tag{1}
$$

## 2.2 Structure

We want to study the structure of functions on sets. Their study in total generality is extremely difficult, so we analyze case-by-case. We begin by analyzing the invariant case when $\check { \mathfrak { X } }$ is a countable set and $\mathcal { V } = \mathbb { R }$ , where the next theorem characterizes its structure.

Theorem 2 Afunction $f ( X )$ operating on a set X having elementsfrom a countable universe, is a valid setfunction, i.e., invariant to the permutation ofinstances in $X , i f f$ it can be decomposed in the form $\textstyle \rho \left( \sum _ { x \in X } \phi ( x ) \right)$ ,for suitable transformations φ and $\rho .$

The extension to case when $\mathfrak { X }$ is uncountable, like ${ \mathfrak { X } } = \mathbb { R }$ , we could only prove that $f ( X ) =$ $\textstyle \rho \left( \sum _ { x \in X . } \phi ( x ) \right)$ holds for sets of fixed size. The proofs and difficulties in handling the uncountable case, are discussed in Appendix A. However, we still conjecture that exact equality holds in general.

Next, we analyze the equivariant case when $\mathfrak { X } = \mathcal { Y } = \mathbb { R }$ and f is restricted to be a neural network layer. The standard neural network layer is represented as $\mathbf { f } _ { \ominus } ( \mathbf { x } ) = \pmb { \sigma } ( \ominus \mathbf { x } )$ where $\Theta \in \mathbb { R } ^ { M \times M }$ is the weight vector and $\sigma : \mathbb { R }  \mathbb { R }$ is a nonlinearity such as sigmoid function. The following lemma states the necessary and sufficient conditions for permutation-equivariance in this type of function.

Lemma 3 The function $\mathbf { f } _ { \ominus } : \mathbb { R } ^ { M } \to \mathbb { R } ^ { M }$ defined above is permutation equivariant iff all the offdiagonal elements of Θ are tied together and all the diagonal elements are equal as well. That is,

$\Theta = \lambda \mathbf { I } + \gamma \left( \mathbf { 1 1 } ^ { \mathsf { T } } \right) \quad \quad \lambda , \gamma \in \mathbb { R } \quad \mathbf { 1 } = [ 1 , \ldots , 1 ] ^ { \mathsf { T } } \in \mathbb { R } ^ { M } \quad \quad \mathbf { I } \in \mathbb { R } ^ { M \times M } i s$ the identity matrix

This result can be easily extended to higher dimensions, $i . e . , \mathfrak { X } = \mathbb { R } ^ { d }$ when $\lambda , \gamma$ can be matrices.

## 2.3 Related Results

The general form of Theorem 2 is closely related with important results in different domains. Here, we quickly review some of these connections.

de Finetti theorem. A related concept is that of an exchangeable model in Bayesian statistics, It is backed by deFinetti’s theorem which states that any exchangeable model can be factored as

$$
p ( X | \alpha , M _ { 0 } ) = \int \mathrm { d } \theta \left[ \prod _ { m = 1 } ^ { M } p ( x _ { m } | \theta ) \right] p ( \theta | \alpha , M _ { 0 } ) ,\tag{2}
$$

where θ is some latent feature and $\alpha , M _ { 0 }$ are the hyper-parameters of the prior. To see that this fits into our result, let us consider exponential families with conjugate priors, where we can analytically calculate the integral of (2). In this special case $p ( x | \theta ) = \mathrm { e x p } \overset { \cdot } { ( } \langle \phi ( \overset { \cdot } { x } ) , \theta \rangle - g ( \theta ) )$ and $p ( \theta | \alpha , \mathbf { \bar { \Lambda } } M _ { 0 } ) \stackrel { - } { = }$ $\mathrm { e x p } \left( \langle \theta , \alpha \rangle - M _ { 0 } ^ { - } g ( \theta ) - h ( \alpha , M _ { 0 } ) \right)$ . Now if we marginalize out θ, we get a form which looks exactly like the one in Theorem 2

$$
p ( X | \alpha , M _ { 0 } ) = \exp \left( h \left( \alpha + \sum _ { m } \phi ( x _ { m } ) , M _ { 0 } + M \right) - h ( \alpha , M _ { 0 } ) \right) .\tag{3}
$$

Representer theorem and kernel machines. Support distribution machines use $\begin{array} { r l } { f ( p ) } & { { } = } \end{array}$ $\textstyle \sum _ { i } ^ { \bullet } \alpha _ { i } y _ { i } K ( p _ { i } , p ) + b$ as the prediction function [8, 10], where $p _ { i } , p$ are distributions and $\alpha _ { i } , b \in \mathbb { R }$ In practice, the $p _ { i } , p$ distributions are never given to us explicitly, usually only i.i.d. sample sets are available from these distributions, and therefore we need to estimate kernel $\check { K } ( p , q )$ using these samples. A popular approach is to use $\begin{array} { r } { \hat { K } ( p , q ) = \frac { 1 } { M M ^ { \prime } } \sum _ { i , j } k ( x _ { i } , y _ { j } ) } \end{array}$ , where k is another kernel operating on the samples $\{ x _ { i } \} _ { i = 1 } ^ { M } \sim p$ and $\{ y _ { j } \} _ { j = 1 } ^ { M ^ { \prime } } \sim q .$ . Now, these prediction functions can be seen fitting into the structure of our Theorem.

Spectral methods. A consequence of the polynomial decomposition is that spectral methods [11] can be viewed as a special case of the mapping $\rho \circ \phi ( X )$ : in that case one can compute polynomials, usually only up to a relatively low degree (such as $k \stackrel { . } { = } 3 )$ , to perform inference about statistical properties of the distribution. The statistics are exchangeable in the data, hence they could be represented by the above map.

## 3 Deep Sets

## 3.1 Architecture

Invariant model. The structure of permutation invariant functions in Theorem 2 hints at a general strategy for inference over sets of objects, which we call DeepSets. Replacing $\phi$ and $\rho$ by universal approximators leaves matters unchanged, since, in particular, φ and $\rho$ can be used to approximate arbitrary polynomials. Then, it remains to learn these approximators, yielding in the following model:

• Each instance $x _ { m }$ is transformed (possibly by several layers) into some representation $\phi ( x _ { m } )$

• The representations $\phi ( x _ { m } )$ are added up and the output is processed using the $\rho$ network in the same manner as in any deep network $( e . g .$ . fully connected layers, nonlinearities, etc.).

• Optionally: If we have additional meta-information $z ,$ then the above mentioned networks could be conditioned to obtain the conditioning mapping $\phi ( x _ { m } | z )$

In other words, the key is to add up all representations and then apply nonlinear transformations.

Equivariant model. Our goal is to design neural network layers that are equivariant to the permutations of elements in the input x. Based on Lemma 3, a neural network layer $\mathbf { f } _ { \Theta } ( \mathbf { x } )$ is permutation equivariant if and only if all the off-diagonal elements of Θ are tied together and all the diagonal elements are equal as well, $i . e . , \Theta = \lambda { \bf I } + \gamma ( { \bf 1 1 } ^ { \mathsf { T } } )$ for $\lambda , \gamma \in \mathbb { R }$ . This function is simply a non-linearity applied to a weighted combination of (i) its input Ix and; (ii) the sum of input values $( \mathbf { 1 1 } ^ { \mathsf { T } } ) \mathbf { x }$ . Since summation does not depend on the permutation, the layer is permutation-equivariant. We can further manipulate the operations and parameters in this layer to get other variations, $e . g . \mathrm { i }$

$$
\mathbf { f } ( \mathbf { x } ) \doteq \pmb { \sigma } \left( \lambda \mathbf { I } \mathbf { x } + \gamma \mathrm { m a x p o o l } ( \mathbf { x } ) \mathbf { 1 } \right) .\tag{4}
$$

where the maxpooling operation over elements of the set (similar to sum) is commutative. In practice, this variation performs better in some applications. This may be due to the fact that for $\lambda = \gamma$ , the input to the non-linearity is max-normalized. Since composition of permutation equivariant functions is also permutation equivariant, we can build DeepSets by stacking such layers.

## 3.2 Other Related Works

Several recent works study equivariance and invariance in deep networks w.r.t. general group of transformations [12–14]. For example, [15] construct deep permutation invariant features by pairwise coupling of features at the previous layer, where $f _ { i , j } ( [ x _ { i } ^ { \cdot } , \stackrel { \cdot } { x } _ { j } ] ) \doteq [ | x _ { i } - x _ { j } | , x _ { i } + x _ { j } ]$ is invariant to transposition of i and j. Pairwise interactions within sets have also been studied in [16, 17]. [18] approach unordered instances by finding “good” orderings.

The idea of pooling a function across set-members is not new. In [19], pooling was used binary classification task for causality on a set of samples. [20] use pooling across a panoramic projection of 3D object for classification, while [21] perform pooling across multiple views. [22] observe the invariance of the payoff matrix in normal form games to the permutation of its rows and columns (i.e. player actions) and leverage pooling to predict the player action. The need of permutation equivariance also arise in deep learning over sensor networks and multi-agent setings, where a special case of Lemma 3 has been used as the architecture [23].

In light of these related works, we would like to emphasize our novel contributions: (i) the universality result of Theorem 2 for permutation invariance that also relates DeepSets to other machine learning techniques, see Sec. 3; (ii) the permutation equivariant layer of (4), which, according to Lemma 3 identifies necessary and sufficient form of parameter-sharing in a standard neural layer and; (iii) novel application settings that we study next.

![](images/8321de893c3a68be8e24821959830e8a3cf77ad24683b9dfae3176368edcd2f1.jpg)

![](images/2ff3df9fa03f4b0c239f8fe106c6b68b4c7fcf2ca4c2dcb9d8d11920fa94990e.jpg)

![](images/f99417ee6cf36b2bfc3e4e49fa5a65c48f82d4f2294f79487d4681de79614fbc.jpg)

![](images/4732eca4664ada35d67e8cd2c99fbbab5cacee312d8942aa8a5f69e5c22f6c00.jpg)

![](images/24587d64cc82d01e725090c18cf0dc718c920c08cbd4b232f12031bd08c016ce.jpg)

![](images/9995d7f98e8e234fa5075548faabd3b83e04963e1a98f7cdab1631bad79fa85e.jpg)  
(a) Entropy estimation for rotated of 2d Gaussian  
(b) Mutual information estimation by varying correlation

![](images/99becb219e40a164c6e85c77fd908ad6918124b2979ac21917e4a5b6d2c91523.jpg)  
(c) Mutual information estimation by varying rank-1 strength

![](images/0cbd0e21416b2a92aa59136de24f0725f685ea9804d50bedd16a53823c995c37.jpg)  
(d) Mutual information on 32d random covariance matrices  
Figure 1: Population statistic estimation: Top set of figures, show prediction of DeepSets vs SDM for $N = 2 ^ { 1 0 }$ case. Bottom set of figures, depict the mean squared error behavior as number of sets is increased. SDM has lower error for small N and DeepSets requires more data to reach similar accuracy. But for high dimensional problems DeepSets easily scales to large number of examples and produces much lower estimation error. Note that the $N \times N$ matrix inversion in SDM makes it prohibitively expensive for $N > 2 ^ { 1 4 } = 1 6 3 8 4$

## 4 Applications and Empirical Results

We present a diverse set of applications for DeepSets. For the supervised setting, we apply DeepSets to estimation of population statistics, sum of digits and classification of point-clouds, and regression with clustering side-information. The permutation-equivariant variation of DeepSets is applied to the task of outlier detection. Finally, we investigate the application of DeepSets to unsupervised set-expansion, in particular, concept-set retrieval and image tagging. In most cases we compare our approach with the state-of-the art and report competitive results.

## 4.1 Set Input Scalar Response

## 4.1.1 Supervised Learning: Learning to Estimate Population Statistics

In the first experiment, we learn entropy and mutual information of Gaussian distributions, without providing any information about Gaussianity to DeepSets. The Gaussians are generated as follows:

• Rotation: We randomly chose a $2 \times 2$ covariance matrix $\Sigma ,$ , and then generated N sample sets from $\mathcal { N } ( 0 , R ( \alpha ) \Sigma R ( \alpha ) ^ { T } )$ of size $M = \left\lceil 3 0 0 - 5 0 0 \right\rceil$ for $N$ random values of $\alpha \in [ 0 , \pi ]$ . Our goal was to learn the entropy of the marginal distribution of first dimension. $R ( \alpha )$ is the rotation matrix.

• Correlation: We randomly chose a $d \times d$ covariance matrix Σ for $d = 1 6 .$ , and then generated $N$ sample sets from $\mathcal { N } ( \bar { 0 } , [ \Sigma , \alpha \Sigma ; \alpha \Sigma , \Sigma ] )$ ) of size $M = \mathrm { [ 3 0 0 - 5 0 0 ] }$ for N random values of $\alpha \in ( - \bar { 1 } , 1 )$ . Goal was to learn the mutual information of among the first d and last d dimension.

• Rank 1: We randomly chose $v \in \mathbb { R } ^ { 3 2 }$ and then generated a sample sets from $\mathcal { N } ( 0 , I + \lambda v v ^ { T } )$ of size $M = \left\lceil 3 0 0 - 5 0 0 \right\rceil$ for N random values of $\lambda \in ( 0 , 1 )$ . Goal was to learn the mutual information.

• Random: We chose N random $l \times d$ covariance matrices Σ for $d = 3 2$ , and using each, generated a sample set from $\mathcal { N } ( 0 , \Sigma )$ of size $M = [ 3 0 0 - 5 0 0 ]$ ]. Goal was to learn the mutual information.

We train using $L _ { 2 }$ loss with a DeepSets architecture having 3 fully connected layers with ReLU activation for both transformations φ and $\rho .$ We compare against Support Distribution Machines (SDM) using a RBF kernel [10], and analyze the results in Fig. 1.

## 4.1.2 Sum of Digits

Next, we compare to what happens if our set data is treated as a sequence. We consider the task of finding sum of a given set of digits. We consider two variants of this experiment:

Text. We randomly sample a subset of maximum $M = 1 0$ digits from this dataset to build $1 0 0 k \ \mathrm { ^ { * } s e t s ^ { , * } }$ of training images, where the setlabel is sum of digits in that set. We test against sums of M digits, for M starting from 5 all the way up to 100 over another 100k examples.

![](images/36a46432b7a1eef727f2b1f0450cde0b817e119e38ff6e0450b47493123757f0.jpg)

![](images/5263c8640791bf499f124c0f097b434b37a1b7ca6b0b2fce317c828b49aab947.jpg)  
Figure 2: Accuracy of digit summation with text (left) and image $( r i g h t )$ inputs. All approaches are trained on tasks of length 10 at most, tested on examples of length up to 100. We see that DeepSets generalizes better.

Image. MNIST8m [24] contains 8 million instances of $2 8 \times 2 8$ grey-scale stamps of digits in $\{ 0 , \ldots , 9 \}$ . We randomly sample a subset of maximum $M = 1 0$ images from this dataset to build $\tilde { N } = 1 0 0 \acute { k }$ “sets” of training and 100k sets of test images, where the set-label is the sum of digits in that set (i.e. individual labels per image is unavailable). We test against sums of M images of MNIST digits, for M starting from 5 all the way up to 50.

We compare against recurrent neural networks – LSTM and GRU. All models are defined to have similar number of layers and parameters. The output of all models is a scalar, predicting the sum of N digits. Training is done on tasks of length 10 at most, while at test time we use examples of length up to 100. The accuracy, i.e. exact equality after rounding, is shown in Fig. 2. DeepSets generalize much better. Note for image case, the best classification error for single digit is around $p = 0 . 0 1$ for MNIST8m, so in a collection of $N$ of images at least one image will be misclassified is $1 - ( 1 - p ) ^ { N }$ which is 40% for $N = 5 0$ . This matches closely with observed value in Fig. 2(b).

## 4.1.3 Point Cloud Classification

A point-cloud is a set of low-dimensional vectors. This type of data is frequently encountered in various applications like robotics, vision, and cosmology. In these applications, existing methods often convert the point-cloud data to voxel or mesh representation as a preprocessing step, $e . g .$ [26, 29, 30]. Since the output of many range sensors, such as LiDAR, is in the form of pointcloud, direct application of deep learning methods to point-cloud is highly desirable. Moreover, it is easy and cheaper to apply transformations, such as rotation and translation, when working with point-clouds than voxelized 3D objects.

As point-cloud data is just a set of points, we can use DeepSets to classify point-cloud representation of a subset of ShapeNet objects [31], called ModelNet40 [25]. This subset consists of 3D representation of 9,843 training and 2,468

<table><tr><td>Model</td><td>Instance Size</td><td>Representation</td><td>Accuracy</td></tr><tr><td>3DShapeNets [25]</td><td> $3 0 ^ { 3 }$ </td><td>voxels (using convo- lutional deep belief net)</td><td>77%</td></tr><tr><td>VoxNet [26]</td><td> $3 2 ^ { 3 }$ </td><td>voxels (voxels from point-cloud  $+ \phantom { + } 3 \mathrm { D }$  CNN)</td><td>83.10%</td></tr><tr><td>MVCNN [21]</td><td>164×164×  $1 2$ </td><td>multi-vew images (2D CNN + view- pooling)</td><td>90.1%</td></tr><tr><td>VRN Ensemble [27]</td><td> $3 2 ^ { 3 }$ </td><td>voxels (3D CNN, variational autoen- coder)</td><td>95.54%</td></tr><tr><td>3D GAN [28]</td><td> $6 4 ^ { 3 }$ </td><td>voxels (3D CNN, generative adversar-</td><td>83.3%</td></tr><tr><td>DeepSets</td><td> $\mathbf { 5 0 0 0 } \times \mathbf { 3 }$ </td><td>ial training) point-cloud</td><td> $9 0 \pm . 3 \%$ </td></tr><tr><td>DeepSets</td><td> $\mathbf { 1 0 0 \times 3 }$ </td><td>point-cloud</td><td> $8 2 \pm 2 \%$ </td></tr></table>

Table 1: Classification accuracy and the representationsize used by different methods on the ModelNet40.

test instances belonging to 40 classes of objects. We produce point-clouds with 100, 1000 and 5000 particles each (x, y, z-coordinates) from the mesh representation of objects using the point-cloudlibrary’s sampling routine [32]. Each set is normalized by the initial layer of the deep network to have zero mean (along individual axes) and unit (global) variance. Tab. 1 compares our method using three permutation equivariant layers against the competition; see Appendix H for details.

## 4.1.4 Improved Red-shift Estimation Using Clustering Information

An important regression problem in cosmology is to estimate the red-shift of galaxies, corresponding to their age as well as their distance from us [33] based on photometric observations. One way to estimate the red-shift from photometric observations is using a regression model [34] on the galaxy clusters. The prediction for each galaxy does not change by permuting the members of the galaxy cluster. Therefore, we can treat each galaxy cluster as $\mathrm { { \bf { a } } } \ \tilde { \ s e t ^ { 3 } }$ and use DeepSets to estimate the individual galaxy red-shifts. See Appendix G for more details.

For each galaxy, we have 17 photometric features from the redMaPPer galaxy cluster catalog [35] that contains photometric readings for 26,111 red galaxy clusters. Each galaxy-cluster in this catalog has between $\sim 2 0 - 3 0 0$ galaxies – i.e. $\mathbf { x } \in \mathbb { R } ^ { N ( c ) \times 1 7 }$ , where $N ( c )$ is the cluster-size. The catalog also provides accurate spectroscopic red-shift estimates for a subset of these galaxies.

<table><tr><td>Method</td><td>Scatter</td></tr><tr><td>MLP</td><td>0.026</td></tr><tr><td>redMaPPer</td><td>0.025</td></tr><tr><td>DeepSets</td><td>0.023</td></tr></table>

Table 2: Red-shift experiment. Lower scatter is better.

We randomly split the data into 90% training and 10% test clusters, and minimize the squared loss of the prediction for available spectroscopic red-shifts. As it is customary in cosmology literature, we report the average scatter $\frac { | z _ { \mathrm { s p e c } } - z | } { 1 + z _ { \mathrm { s p e c } } }$ , where $z _ { \mathrm { s p e c } }$ is the accurate spectroscopic measurement and z is a photometric estimate in Tab. 2.

<table><tr><td rowspan="3">Method</td><td colspan="5">LDA-1k (Vocab = 17k)</td><td colspan="5">LDA-3k (Vocab = 38k)</td><td colspan="5">LDA-5k (Vocab = 61k)</td></tr><tr><td colspan="2">Recall (%) @10</td><td rowspan="2">@1k</td><td rowspan="2"></td><td rowspan="2">MRR Med.</td><td rowspan="2">@10</td><td rowspan="2">Recall (%) @100</td><td rowspan="2">@1k</td><td rowspan="2">MRR</td><td rowspan="2">Med.</td><td rowspan="2">@10 @100</td><td rowspan="2">Recall (%)</td><td rowspan="2"></td><td rowspan="2">@1k</td><td rowspan="2">MRR Med.</td></tr><tr><td>@100</td><td></td></tr><tr><td>Random</td><td>0.06</td><td>0.6</td><td>5.9</td><td>0.001</td><td>8520</td><td>0.02</td><td>0.2</td><td>2.6</td><td>0.000</td><td>28635</td><td>0.01</td><td>0.2</td><td>1.6</td><td>0.000</td><td>30600</td></tr><tr><td>Bayes Set</td><td>1.69</td><td>11.9</td><td>37.2</td><td>0.007</td><td>2848</td><td>2.01</td><td>14.5</td><td>36.5</td><td>0.008</td><td>3234</td><td>1.75</td><td>12.5</td><td>34.5</td><td>0.007</td><td>3590</td></tr><tr><td>w2v Near</td><td>6.00</td><td>28.1</td><td>54.7</td><td>0.021</td><td>641</td><td>4.80</td><td>21.2</td><td>43.2</td><td>0.016</td><td>2054</td><td>4.03</td><td>16.7</td><td>35.2</td><td>0.013</td><td>6900</td></tr><tr><td>NN-max</td><td>4.78</td><td>22.5</td><td>53.1</td><td>0.023</td><td>779</td><td>5.30</td><td>24.9</td><td>54.8</td><td>0.025</td><td>672</td><td>4.72</td><td>21.4</td><td>47.0</td><td>0.022</td><td>1320</td></tr><tr><td>NN-sum-con</td><td>4.58</td><td>19.8</td><td>48.5</td><td>0.021</td><td>1110</td><td>5.81</td><td>27.2</td><td>60.0</td><td>0.027</td><td>453</td><td>4.87</td><td>23.5</td><td>53.9</td><td>0.022</td><td>731</td></tr><tr><td>NN-max-con</td><td>3.36</td><td>16.9</td><td>46.6</td><td>0.018</td><td>1250</td><td>5.61</td><td>25.7</td><td>57.5</td><td>0.026</td><td>570</td><td>4.72</td><td>22.0</td><td>51.8</td><td>0.022</td><td>877</td></tr><tr><td>DeepSets</td><td>5.53</td><td>24.2</td><td>54.3</td><td>0.025</td><td>696</td><td>6.04</td><td>28.5</td><td>60.7</td><td>0.027</td><td>426</td><td>5.54</td><td>26.1</td><td>55.5</td><td>0.026</td><td>616</td></tr></table>

Table 3: Results on Text Concept Set Retrieval on LDA-1k, LDA-3k, and LDA-5k. Our DeepSets model outperforms other methods on LDA-3k and LDA-5k. However, all neural network based methods have inferior performance to w2v-Near baseline on LDA-1k, possibly due to small data size. Higher the better for recall@k and mean reciprocal rank (MRR). Lower the better for median rank (Med.)

## 4.2 Set Expansion

In the set expansion task, we are given a set of objects that are similar to each other and our goal is to find new objects from a large pool of candidates such that the selected new objects are similar to the query set. To achieve this one needs to reason out the concept connecting the given set and then retrieve words based on their relevance to the inferred concept. It is an important task due to wide range of potential applications including personalized information retrieval, computational advertisement, tagging large amounts of unlabeled or weakly labeled datasets.

Going back to de Finetti’s theorem in Sec. 3.2, where we consider the marginal probability of a set of observations, the marginal probability allows for very simple metric for scoring additional elements to be added to X. In other words, this allows one to perform set expansion via the following score

$$
s ( x | X ) = \log p ( X \cup \{ x \} | \alpha ) - \log p ( X | \alpha ) p ( \{ x \} | \alpha )\tag{5}
$$

Note that $s ( x | X )$ is the point-wise mutual information between x and X. Moreover, due to exchangeability, it follows that regardless of the order of elements we have M

$$
S ( X ) = \sum _ { m } s \left( x _ { m } \vert \left\{ x _ { m - 1 } , \ldots x _ { 1 } \right\} \right) = \log p ( X \vert \alpha ) - \sum _ { m = 1 } \log p ( \left\{ x _ { m } \right\} \vert \alpha )\tag{6}
$$

When inferring sets, our goal is to find set completions $\{ x _ { m + 1 } , \hdots x _ { M } \}$ for an initial set of query terms $\{ x _ { 1 } , \ldots , x _ { m } \}$ , such that the aggregate set is coherent. This is the key idea of the Bayesian Set algorithm [36] (details in Appendix D). Using DeepSets, we can solve this problem in more generality as we can drop the assumption of data belonging to certain exponential family.

For learning the score $s ( x | X )$ , we take recourse to large-margin classification with structured loss functions [37] to obtain the relative loss objective $l ( x , x ^ { \prime } | \mathbf { \tilde { \boldsymbol { X } } } ) = \mathbf { \tilde { m a x } } \big ( 0 , s ( x ^ { \prime } | \boldsymbol { X } ) - s ( x | \boldsymbol { X } ) + \Delta ( x , x ^ { \prime } ) \big )$ In other words, we want to ensure that $s ( x | X ) \stackrel { \cdot } { \geq } s ( x ^ { \prime } | X ) + \Delta ( x , \stackrel { \cdot } { x ^ { \prime } } )$ whenever x should be added and $x ^ { \prime }$ should not be added to X.

Conditioning. Often machine learning problems do not exist in isolation. For example, task like tag completion from a given set of tags is usually related to an object z, for example an image, that needs to be tagged. Such meta-data are usually abundant, e.g. author information in case of text, contextual data such as the user click history, or extra information collected with LiDAR point cloud.

Conditioning graphical models with meta-data is often complicated. For instance, in the Beta-Binomial model we need to ensure that the counts are always nonnegative, regardless of z. Fortunately, DeepSets does not suffer from such complications and the fusion of multiple sources of data can be done in a relatively straightforward manner. Any of the existing methods in deep learning, including feature concatenation by averaging, or by max-pooling, can be employed. Incorporating these metadata often leads to significantly improved performance as will be shown in experiments; Sec. 4.2.2.

## 4.2.1 Text Concept Set Retrieval

In text concept set retrieval, the objective is to retrieve words belonging to a ‘concept’ or ‘cluster’, given few words from that particular concept. For example, given the set of words {tiger, lion, cheetah}, we would need to retrieve other related words like jaguar, puma, etc, which belong to the same concept of big cats. This task of concept set retrieval can be seen as a set completion task conditioned on the latent semantic concept, and therefore our DeepSets form a desirable approach.

Dataset. We construct a large dataset containing sets of $N _ { T } = 5 0$ related words by extracting topics from latent Dirichlet allocation [38, 39], taken out-of-the-box<sup>1</sup>. To compare across scales, we consider three values of $\boldsymbol { k } = \{ 1 k , 3 k , 5 k \}$ giving us three datasets LDA-1k, LDA-3k, and LDA-5k, with corresponding vocabulary sizes of 17k, 38k, and 61k.

Methods. We learn this using a margin loss with a DeepSets architecture having 3 fully connected layers with ReLU activation for both transformations φ and ρ. Details of the architecture and training are in Appendix E. We compare to several baselines: (a) Random picks a word from the vocabulary uniformly at random. (b) Bayes Set [36]. (c) w2v-Near computes the nearest neighbors in the word2vec [40] space. Note that both Bayes Set and w2v NN are strong baselines. The former runs Bayesian inference using Beta-Binomial conjugate pair, while the latter uses the powerful 300 dimensional word2vec trained on the billion word GoogleNews corpus<sup>2</sup>. (d) NN-max uses a similar architecture as our DeepSets but uses max pooling to compute the set feature, as opposed to sum pooling. (e) NN-max-con uses max pooling on set elements but concatenates this pooled representation with that of query for a final set feature. (f) NN-sum-con is similar to NN-max-con but uses sum pooling followed by concatenation with query representation.

Evaluation. We consider the standard retrieval metrics – recall@K, median rank and mean reciprocal rank, for evaluation. To elaborate, recall@K measures the number of true labels that were recovered in the top K retrieved words. We use three values of $\mathbf { K } = \{ 1 0 , 1 0 0 , 1 k \}$ . The other two metrics, as the names suggest, are the median and mean of reciprocals of the true label ranks, respectively. Each dataset is split into TRAIN (80%), VAL (10%) and TEST (10%). We learn models using TRAIN and evaluate on TEST, while VAL is used for hyperparameter selection and early stopping.

Results and Observations. As seen in Tab. 3: (a) Our DeepSets model outperforms all other approaches on LDA-3k and LDA-5k by any metric, highlighting the significance of permutation invariance property. (b) On LDA-1k, our model does not perform well when compared to w2v-Near. We hypothesize that this is due to small size of the dataset insufficient to train a high capacity neural network, while w2v-Near has been trained on a billion word corpus. Nevertheless, our approach comes the closest to w2v-Near amongst other approaches, and is only 0.5% lower by Recall@10.

## 4.2.2 Image Tagging

We next experiment with image tagging, where the task is to retrieve all relevant tags corresponding to an image. Images usually have only a subset of relevant tags, therefore predicting other tags can help enrich information that can further be leveraged in a downstream supervised task. In our setup, we learn to predict tags by conditioning DeepSets on the image, i.e., we train to predict a partial set of tags from the image and remaining tags. At test time, we predict tags from the image alone.

Datasets. We report results on the following three datasets - ESPGame, IAPRTC-12.5 and our in-house dataset, COCO-Tag. We refer the reader to Appendix F, for more details about datasets.

<table><tr><td rowspan="2">Method</td><td colspan="4">ESP game</td><td colspan="4">IAPRTC-12.5</td></tr><tr><td>P</td><td>R</td><td></td><td>F1 N+</td><td>P</td><td></td><td>R F1 N+</td><td></td></tr><tr><td rowspan="5">Least Sq. MBRM JEC FastTag Least Šq.(D)</td><td>35</td><td></td><td></td><td>5 19 25 215</td><td></td><td></td><td>40 19 26 198</td><td></td></tr><tr><td></td><td></td><td></td><td>18 19 18 209</td><td></td><td></td><td></td><td>24 23 23 223</td></tr><tr><td></td><td></td><td></td><td>24 19 21 222</td><td></td><td></td><td></td><td>29 19 23 211</td></tr><tr><td></td><td></td><td></td><td>46 22 30 247</td><td></td><td></td><td></td><td>47 26 34 280</td></tr><tr><td></td><td></td><td></td><td>44 32 37 232</td><td></td><td></td><td></td><td>46 30 36 218</td></tr><tr><td>FastTag(D)</td><td></td><td></td><td></td><td>44 32 37 229</td><td></td><td></td><td></td><td>46 33 38 254</td></tr><tr><td>DeepSets</td><td></td><td></td><td></td><td>39 34 36 246</td><td></td><td></td><td>42 31 36 247</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 4: Results of image tagging on ESPgame and IAPRTC-12.5 datasets. Performance of our DeepSets approach is roughly similar to the best competing approaches, except for precision. Refer text for more details. Higher the better for all metrics – precision (P), recall (R), f1 score (F1), and number of non-zero recall tags (N+).

Methods. The setup for DeepSets to tag images is similar to that described in Sec. 4.2.1. The only difference being the conditioning on the image features, which is

concatenated with the set feature obtained from pooling individual element representations.

Baselines. We perform comparisons against several baselines, previously reported in [41]. Specifically, we have Least Sq., a ridge regression model, MBRM [42], JEC [43] and FastTag [41]. Note that these methods do not use deep features for images, which could lead to an unfair comparison. As there is no publicly available code for MBRM and JEC, we cannot get performances of these models with Resnet extracted features. However, we report results with deep features for FastTag and Least Sq., using code made available by the authors <sup>3</sup>.

Evaluation. For ESPgame and IAPRTC-12.5, we follow the evaluation metrics as in [44]–precision (P), recall (R), F1 score (F1), and number of tags with non-zero recall (N+). These metrics are evaluate for each tag and the mean is reported (see [44] for further details). For COCO-Tag, however, we use recall@K for three values of K = {10, 100, 1000}, along with median rank and mean reciprocal rank (see evaluation in Sec. 4.2.1 for metric details).

![](images/504bafe50afcbefe1aa8b4e9e12f00f3f3b2e1c286a38f3e5c9b0f16693e9042.jpg)  
Figure 3: Each row shows a set, constructed from CelebA dataset, such that all set members except for an outlier, share at least two attributes (on the right). The outlier is identified with a red frame. The model is trained by observing examples of sets and their anomalous members, without access to the attributes. The probability assigned to each member by the outlier detection network is visualized using a red bar at the bottom of each image. The probabilities in each row sum to one.

Results and Observations. Tab. 4 shows results of image tagging on ESPgame and IAPRTC-12.5, and Tab. 5 on COCO-Tag. Here are the key observations from Tab. 4: (a) performance of our DeepSets model is comparable to the best approaches on all metrics but precision, (b) our recall beats the best approach by 2% in ESPgame. On further investigation, we found that the DeepSets model retrieves more relevant tags, which are not present in list of ground truth tags due to a limited 5 tag annotation. Thus, this takes a toll on precision while gaining on recall, yet yielding improvement on F1. On the larger and richer COCO-Tag, we see that the DeepSets approach outperforms other methods comprehensively, as expected. Qualitative examples are in Appendix F.

<table><tr><td>Method</td><td colspan="3">Recall</td><td>MRR Med.</td><td></td></tr><tr><td></td><td>@10</td><td>@100</td><td>@1k</td><td></td><td></td></tr><tr><td>w2v NN (blind) DeepSets (blind)</td><td>5.6 9.0</td><td>20.0 39.2</td><td>54.2 71.3</td><td>0.021 0.044</td><td>823 310</td></tr><tr><td></td><td></td><td></td><td></td><td>0.131</td><td>28</td></tr><tr><td>DeepSets</td><td>31.4</td><td>73.4</td><td>95.3</td><td></td><td></td></tr></table>

Table 5: Results on COCO-Tag dataset. Clearly, DeepSets outperforms other baselines significantly. Higher the better for recall@K and mean reciprocal rank (MRR). Lower the better for median rank (Med).

## 4.3 Set Anomaly Detection

The objective here is to find the anomalous face in each set, simply by observing examples and without any access to the attribute values. CelebA dataset [45] contains 202,599 face images, each annotated with 40 boolean attributes. We build N = 18, 000 sets of 64 × 64 stamps, using these attributes each containing M = 16 images (on the training set) as follows: randomly select 2 attributes, draw 15 images having those attributes, and a single target image where both attributes are absent. Using a similar procedure we build sets on the test images. No individual person‘s face appears in both train and test sets. Our deep neural network consists of 9 2D-convolution and max-pooling layers followed by 3 permutation-equivariant layers, and finally a softmax layer that assigns a probability value to each set member (Note that one could identify arbitrary number of outliers using a sigmoid activation at the output). Our trained model successfully finds the anomalous face in 75% of test sets. Visually inspecting these instances suggests that the task is non-trivial even for humans; see Fig. 3.

As a baseline, we repeat the same experiment by using a set-pooling layer after convolution layers, and replacing the permutation-equivariant layers with fully connected layers of same size, where the final layer is a 16-way softmax. The resulting network shares the convolution filters for all instances within all sets, however the input to the softmax is not equivariant to the permutation of input images. Permutation equivariance seems to be crucial here as the baseline model achieves a training and test accuracy of ∼ 6.3%; the same as random selection. See Appendix I for more details.

## 5 Summary

In this paper, we develop DeepSets, a model based on powerful permutation invariance and equivariance properties, along with the theory to support its performance. We demonstrate the generalization ability of DeepSets across several domains by extensive experiments, and show both qualitative and quantitative results. In particular, we explicitly show that DeepSets outperforms other intuitive deep networks, which are not backed by theory (Sec. 4.2.1, Sec. 4.1.2). Last but not least, it is worth noting that the state-of-the-art we compare to is a specialized technique for each task, whereas our one model, i.e., DeepSets, is competitive across the board.

## References

[1] B. Poczos, A. Rinaldo, A. Singh, and L. Wasserman. Distribution-free distribution regression. In International Conference on AI and Statistics (AISTATS), JMLR Workshop and Conference Proceedings, 2013. pages

[2] I. Jung, M. Berges, J. Garrett, and B. Poczos. Exploration and evaluation of ar, mpca and kl anomaly detection techniques to embankment dam piezometer data. Advanced Engineering Informatics, 2015. pages

[3] M. Ntampaka, H. Trac, D. Sutherland, S. Fromenteau, B. Poczos, and J. Schneider. Dynamical mass measurements of contaminated galaxy clusters using machine learning. The Astrophysical Journal, 2016. URL http://arxiv.org/abs/1509.05409. pages

[4] M. Ravanbakhsh, J. Oliva, S. Fromenteau, L. Price, S. Ho, J. Schneider, and B. Poczos. Estimating cosmological parameters from the dark matter distribution. In International Conference on Machine Learning (ICML), 2016. pages

[5] J. Oliva, B. Poczos, and J. Schneider. Distribution to distribution regression. In International Conference on Machine Learning (ICML), 2013. pages

[6] Z. Szabo, B. Sriperumbudur, B. Poczos, and A. Gretton. Learning theory for distribution regression. Journal ofMachine Learning Research, 2016. pages

[7] K. Muandet, D. Balduzzi, and B. Schoelkopf. Domain generalization via invariant feature representation. In In Proceeding of the 30th International Conference on Machine Learning (ICML 2013), 2013. pages

[8] K. Muandet, K. Fukumizu, F. Dinuzzo, and B. Schoelkopf. Learning from distributions via support measure machines. In In Proceeding of the 26th Annual Conference on Neural Information Processing Systems (NIPS 2012), 2012. pages

[9] Felix A. Faber, Alexander Lindmaa, O. Anatole von Lilienfeld, and Rickard Armiento. Machine learning energies of 2 million elpasolite $\left( a b C _ { 2 } D _ { 6 } \right)$ crystals. Phys. Rev. Lett., 117:135502, Sep 2016. doi: 10.1103/PhysRevLett.117.135502. pages

[10] B. Poczos, L. Xiong, D. Sutherland, and J. Schneider. Support distribution machines, 2012. URL http://arxiv.org/abs/1202.0302. pages

[11] A. Anandkumar, R. Ge, D. Hsu, S. M. Kakade, and M. Telgarsky. Tensor decompositions for learning latent variable models. arXiv preprint arXiv:1210.7559, 2012. pages

[12] Robert Gens and Pedro M Domingos. Deep symmetry networks. In Advances in neural information processing systems, pages 2537–2545, 2014. pages

[13] Taco S Cohen and Max Welling. Group equivariant convolutional networks. arXiv preprint arXiv:1602.07576, 2016. pages

[14] Siamak Ravanbakhsh, Jeff Schneider, and Barnabas Poczos. Equivariance through parametersharing. arXiv preprint arXiv:1702.08389, 2017. pages

[15] Xu Chen, Xiuyuan Cheng, and Stéphane Mallat. Unsupervised deep haar scattering on graphs. In Advances in Neural Information Processing Systems, pages 1709–1717, 2014. pages

[16] Michael B Chang, Tomer Ullman, Antonio Torralba, and Joshua B Tenenbaum. A compositional object-based approach to learning physical dynamics. arXiv preprint arXiv:1612.00341, 2016. pages

[17] Nicholas Guttenberg, Nathaniel Virgo, Olaf Witkowski, Hidetoshi Aoki, and Ryota Kanai. Permutation-equivariant neural networks applied to dynamics prediction. arXiv preprint arXiv:1612.04530, 2016. pages

[18] Oriol Vinyals, Samy Bengio, and Manjunath Kudlur. Order matters: Sequence to sequence for sets. arXiv preprint arXiv:1511.06391, 2015. pages

[19] David Lopez-Paz, Robert Nishihara, Soumith Chintala, Bernhard Schölkopf, and Léon Bottou. Discovering causal signals in images. arXiv preprint arXiv:1605.08179, 2016. pages

[20] Baoguang Shi, Song Bai, Zhichao Zhou, and Xiang Bai. Deeppano: Deep panoramic representation for 3-d shape recognition. IEEE Signal Processing Letters, 22(12):2339–2343, 2015. pages

[21] Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller. Multi-view convolutional neural networks for 3d shape recognition. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 945–953, 2015. pages

[22] Jason S Hartford, James R Wright, and Kevin Leyton-Brown. Deep learning for predicting human strategic behavior. In Advances in Neural Information Processing Systems, pages 2424–2432, 2016. pages

[23] Sainbayar Sukhbaatar, Rob Fergus, et al. Learning multiagent communication with backpropagation. In Neural Information Processing Systems, pages 2244–2252, 2016. pages

[24] Gaëlle Loosli, Stéphane Canu, and Léon Bottou. Training invariant support vector machines using selective sampling. In Léon Bottou, Olivier Chapelle, Dennis DeCoste, and Jason Weston, editors, Large Scale Kernel Machines, pages 301–320. MIT Press, Cambridge, MA., 2007. pages

[25] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1912–1920, 2015. pages

[26] Daniel Maturana and Sebastian Scherer. Voxnet: A 3d convolutional neural network for realtime object recognition. In Intelligent Robots and Systems (IROS), 2015 IEEE/RSJ International Conference on, pages 922–928. IEEE, 2015. pages

[27] Andrew Brock, Theodore Lim, JM Ritchie, and Nick Weston. Generative and discriminative voxel modeling with convolutional neural networks. arXiv preprint arXiv:1608.04236, 2016. pages

[28] Jiajun Wu, Chengkai Zhang, Tianfan Xue, William T Freeman, and Joshua B Tenenbaum. Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. arXiv preprint arXiv:1610.07584, 2016. pages

[29] Siamak Ravanbakhsh, Junier Oliva, Sebastien Fromenteau, Layne C Price, Shirley Ho, Jeff Schneider, and Barnabás Póczos. Estimating cosmological parameters from the dark matter distribution. In Proceedings ofThe 33rd International Conference on Machine Learning, 2016. pages

[30] Hong-Wei Lin, Chiew-Lan Tai, and Guo-Jin Wang. A mesh reconstruction algorithm driven by an intrinsic property of a point cloud. Computer-Aided Design, 36(1):1–9, 2004. pages

[31] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015. pages

[32] Radu Bogdan Rusu and Steve Cousins. 3D is here: Point Cloud Library (PCL). In IEEE International Conference on Robotics and Automation (ICRA), Shanghai, China, May 9-13 2011. pages

[33] James Binney and Michael Merrifield. Galactic astronomy. Princeton University Press, 1998. pages

[34] AJ Connolly, I Csabai, AS Szalay, DC Koo, RG Kron, and JA Munn. Slicing through multicolor space: Galaxy redshifts from broadband photometry. arXiv preprint astro-ph/9508100, 1995. pages

[35] Eduardo Rozo and Eli S Rykoff. redmapper ii: X-ray and sz performance benchmarks for the sdss catalog. The Astrophysical Journal, 783(2):80, 2014. pages

[36] Zoubin Ghahramani and Katherine A Heller. Bayesian sets. In NIPS, volume 2, pages 22–23, 2005. pages

[37] B. Taskar, C. Guestrin, and D. Koller. Max-margin Markov networks. In S. Thrun, L. Saul, and B. Schölkopf, editors, Advances in Neural Information Processing Systems 16, pages 25–32, Cambridge, MA, 2004. MIT Press. pages

[38] Jonathan K. Pritchard, Matthew Stephens, and Peter Donnelly. Inference of population structure using multilocus genotype data. Genetics, 155(2):945–959, 2000. ISSN 0016-6731. URL http://www.genetics.org/content/155/2/945. pages

[39] David M. Blei, Andrew Y. Ng, Michael I. Jordan, and John Lafferty. Latent dirichlet allocation. Journal ofMachine Learning Research, 3:2003, 2003. pages

[40] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pages 3111–3119, 2013. pages

[41] Minmin Chen, Alice Zheng, and Kilian Weinberger. Fast image tagging. In Proceedings of The 30th International Conference on Machine Learning, pages 1274–1282, 2013. pages

[42] S. L. Feng, R. Manmatha, and V. Lavrenko. Multiple bernoulli relevance models for image and video annotation. In Proceedings ofthe 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, CVPR’04, pages 1002–1009, Washington, DC, USA, 2004. IEEE Computer Society. pages

[43] Ameesh Makadia, Vladimir Pavlovic, and Sanjiv Kumar. A new baseline for image annotation. In Proceedings of the 10th European Conference on Computer Vision: Part III, ECCV ’08, pages 316–329, Berlin, Heidelberg, 2008. Springer-Verlag. pages

[44] Matthieu Guillaumin, Thomas Mensink, Jakob Verbeek, and Cordelia Schmid. Tagprop: Discriminative metric learning in nearest neighbor models for image auto-annotation. In Computer Vision, 2009 IEEE 12th International Conference on, pages 309–316. IEEE, 2009. pages

[45] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings ofInternational Conference on Computer Vision (ICCV), 2015. pages

[46] Branko Curgus and Vania Mascioni. Roots and polynomials as homeomorphic spaces. <sup>´</sup> Expositiones Mathematicae, 24(1):81–95, 2006. pages

[47] Boris A Khesin and Serge L Tabachnikov. Arnold: Swimming Against the Tide, volume 86. American Mathematical Society, 2014. pages

[48] Jerrold E Marsden and Michael J Hoffman. Elementary classical analysis. Macmillan, 1993. pages

[49] Nicolas Bourbaki. Eléments de mathématiques: théorie des ensembles, chapitres 1 à 4, volume 1. Masson, 1990. pages

[50] C. A. Micchelli. Interpolation of scattered data: distance matrices and conditionally positive definite functions. Constructive Approximation, 2:11–22, 1986. pages

[51] Luis Von Ahn and Laura Dabbish. Labeling images with a computer game. In Proceedings of the SIGCHI conference on Humanfactors in computing systems, pages 319–326. ACM, 2004. pages

[52] Michael Grubinger. Analysis and evaluation of visual information systems performance, 2007. URL http://eprints.vu.edu.au/1435. Thesis (Ph. D.)–Victoria University (Melbourne, Vic.), 2007. pages

[53] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, pages 740–755. Springer, 2014. pages

[54] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. pages

[55] Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289, 2015. pages

# Appendix: Deep Sets

## A Proofs and Discussion Related to Theorem 2

A function f transforms its domain X into its range Y. Usually, the input domain is a vector space $\mathbb { R } ^ { d }$ and the output response range is either a discrete space, $\mathbf { e . g . \{ 0 , 1 \} }$ in case of classification, or a continuous space R in case of regression.

Now, if the input is a set $X = \{ x _ { 1 } , \dots , x _ { M } \} , x _ { m } \in { \mathfrak { X } } , { \mathrm { i . e . ~ } } { \mathcal { X } } = 2 ^ { \mathfrak { X } }$ , then we would like the response of the function not to depend on the ordering of the elements in the set. In other words,

Property 1 A function $f : 2 ^ { \mathfrak { X } } \to \mathbb { R }$ acting on sets must be permutation invariant to the order of objects in the set, i.e.

$$
f ( \{ x _ { 1 } , . . . , x _ { M } \} ) = f ( \{ x _ { \pi ( 1 ) } , . . . , x _ { \pi ( M ) } \} )\tag{7}
$$

for any permutation π.

Now, roughly speaking, we claim that such functions must have a structure of the form $f ( X ) =$ $\textstyle \rho \left( \sum _ { x \in X } \phi ( x ) \right)$ for some functions $\rho$ and $\phi .$ . Over the next two sections we try to formally prove this structure of the permutation invariant functions.

## A.1 Countable Case

Theorem 2 Assume the elements are countable, i.e. $| { \mathfrak { X } } | < \aleph _ { 0 } .$ . A function $f : 2 ^ { \mathfrak { X } } \to$ R operating on a set X can be a valid setfunction, i.e. it is permutation invariant to the elements in $X$ , ifand only if it can be decomposed in theform $\textstyle \rho \left( \sum _ { x \in X } \phi ( x ) \right)$ ,for suitable transformations $\phi$ and $\rho .$

Proof. Permutation invariance follows from the fact that sets have no particular order, hence any function on a set must not exploit any particular order either. The sufficiency follows by observing that the function $\textstyle \rho \left( \sum _ { x \in X } \phi ( x ) \right)$ satisfies the permutation invariance condition.

To prove necessity, i.e. that all functions can be decomposed in this manner, we begin by noting that there must be a mapping from the elements to natural numbers functions, since the elements are countable. Let this mapping be denoted by $c : { \mathfrak { X } } \to \mathbb { N }$ . Now if we let $\phi ( x ) = 4 ^ { - c ( x ) }$ then $\textstyle \sum _ { x \in X } \phi ( x )$ constitutes an unique representation for every set $X \in 2 ^ { \mathfrak { X } }$ . Now a function $\rho : \mathbb { R }  \mathbb { R }$ can always be constructed such that $\begin{array} { r } { f ( X ) = \rho \left( \sum _ { x \in X } \phi ( x ) \right) } \end{array}$

## A.2 Uncountable Case

The extension to case when X is uncountable, $e . g .  \mathfrak { X } = [ 0 , 1 ]$ , is not so trivial. We could only prove in case of fixed set size, e.g. $\mathcal { X } = [ 0 , 1 ] ^ { M }$ instead of $\mathcal { X } = 2 ^ { \bar { \mathfrak { X } } } = 2 ^ { [ 0 , 1 ] }$ , that any permutation invariant continuous function can be expressed as $\textstyle \rho \left( \sum _ { x \in X } \phi ( x ) \right) . { \mathrm { A l s o } }$ , we show that there is a universal approximator of the same form. These results are discussed below.

To illustrate the uncountable case, we assume a fixed set size of M. Without loss of generality we can let ${ \mathfrak { X } } = [ 0 , 1 ]$ . Then the domain becomes $[ 0 , 1 ] ^ { M } . \mathrm { A l s o }$ , to handle ambiguity due to permutation, we often define the domain to be the set $\mathcal { X } = \{ ( \bar { x } _ { 1 } , . . . , x _ { M } ) \in [ 0 , 1 ] ^ { M } : x _ { 1 } \leq x _ { 2 } \leq \cdot \cdot \cdot \leq x _ { M } \}$ for some ordering of the elements in X.

The proof builds on the famous Newton-Girard formulae which connect moments of a sample set (sum-of-power) to the elementary symmetric polynomials. But first we present some results needed for the proof. The first result establishes that sum-of-power mapping is injective.

Lemma 4 $L e t \mathcal { X } = \{ ( x _ { 1 } , . . . , x _ { M } ) \in [ 0 , 1 ] ^ { M } : x _ { 1 } \leq x _ { 2 } \leq \cdot \cdot \cdot \leq x _ { M } \}$ . The sum-of-power mapping E : X → R<sup>M+1</sup> defined by the coordinatefunctions

$$
Z _ { q } : = E _ { q } ( X ) : = \sum _ { m = 1 } ^ { M } ( x _ { m } ) ^ { q } , \qquad q = 0 , . . . , M .\tag{8}
$$

is injective.

Proof. Suppose for some $u , v \in { \mathcal { X } } ,$ , we have $E ( u ) = E ( v )$ ). We will now show that it must be the case that $u = v$ . Construct two polynomials as follows:

$$
P _ { u } ( x ) = \prod _ { m = 1 } ^ { M } ( x - u _ { m } ) \qquad P _ { v } ( x ) = \prod _ { m = 1 } ^ { M } ( x - v _ { m } )\tag{9}
$$

If we expand the two polynomials we obtain:

$$
\begin{array} { l } { P _ { u } ( x ) = x ^ { M } - a _ { 1 } x ^ { M - 1 } + \cdot \cdot \cdot ( - 1 ) ^ { M - 1 } a _ { M - 1 } x + ( - 1 ) ^ { M } a _ { M } } \\ { P _ { v } ( x ) = x ^ { M } - b _ { 1 } x ^ { M - 1 } + \cdot \cdot \cdot ( - 1 ) ^ { M - 1 } b _ { M - 1 } x + ( - 1 ) ^ { M } b _ { M } } \end{array}\tag{10}
$$

with coefficients being elementary symmetric polynomials in u and v respectively, i.e.

$$
a _ { m } = \sum _ { \substack { 1 \leq j _ { 1 } < j _ { 2 } < \cdots < j _ { m } \leq M } } u _ { j _ { 1 } } u _ { j _ { 2 } } \cdot \cdot \cdot u _ { j _ { m } } \qquad b _ { m } = \sum _ { \substack { 1 \leq j _ { 1 } < j _ { 2 } < \cdots < j _ { m } \leq M } } v _ { j _ { 1 } } v _ { j _ { 2 } } \cdot \cdot \cdot v _ { j _ { m } }\tag{11}
$$

These elementary symmetric polynomials can be uniquely expressed as a function of $E ( u )$ and $E ( v )$ respectively, by Newton-Girard formula. The m-th coefficient is given by the determinant of $m \times m$ matrix having terms from $E ( u )$ and $E ( v )$ respectively:

$$
a _ { m } = \frac { 1 } { m } \operatorname * { d e t } \left( \begin{array} { c c c c c c } { E _ { 1 } ( u ) } & { 1 } & { 0 } & { 0 } & { \cdots } & { 0 } \\ { E _ { 2 } ( u ) } & { E _ { 1 } ( u ) } & { 1 } & { 0 } & { \cdots } & { 0 } \\ { E _ { 3 } ( u ) } & { E _ { 2 } ( u ) } & { E _ { 1 } ( u ) } & { 1 } & { \cdots } & { 0 } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { E _ { m - 1 } ( u ) } & { E _ { m - 2 } ( u ) } & { E _ { m - 3 } ( u ) } & { E _ { m - 4 } ( u ) } & { \cdots } & { 1 } \\ { E _ { m } ( u ) } & { E _ { m - 1 } ( u ) } & { E _ { m - 2 } ( u ) } & { E _ { m - 3 } ( u ) } & { \cdots } & { E _ { 1 } ( u ) } \end{array} \right)\tag{12}
$$

$$
b _ { m } = \frac { 1 } { m } \operatorname * { d e t } \left( \begin{array} { c c c c c c } { E _ { 1 } ( v ) } & { 1 } & { 0 } & { 0 } & { \cdots } & { 0 } \\ { E _ { 2 } ( v ) } & { E _ { 1 } ( v ) } & { 1 } & { 0 } & { \cdots } & { 0 } \\ { E _ { 3 } ( v ) } & { E _ { 2 } ( v ) } & { E _ { 1 } ( v ) } & { 1 } & { \cdots } & { 0 } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { E _ { m - 1 } ( v ) } & { E _ { m - 2 } ( v ) } & { E _ { m - 3 } ( v ) } & { E _ { m - 4 } ( v ) } & { \cdots } & { 1 } \\ { E _ { m } ( v ) } & { E _ { m - 1 } ( v ) } & { E _ { m - 2 } ( v ) } & { E _ { m - 3 } ( v ) } & { \cdots } & { E _ { 1 } ( v ) } \end{array} \right)
$$

Since we assumed $E ( u ) = E ( v )$ implying $[ a _ { 1 } , . . . , a _ { M } ] = [ b _ { 1 } , . . . , b _ { M } ]$ , which in turn implies that the polynomials $P _ { u }$ and $\dot { P _ { v } }$ are the same. Therefore, their roots must be the same, which shows that $u = v$

The second result we borrow from [46] which establishes a homeomorphism between coefficients and roots of a polynomial.

Theorem 5 [46] The function $f : \mathbb { C } ^ { M } \to \mathbb { C } ^ { M }$ , which associates every $a \in \mathbb { C } ^ { M }$ to the multiset of roots, $f ( a ) \in \mathbb { C } ^ { M }$ , ofthe monic polynomialformed using a as the coefficient i.e. $x ^ { M } + a _ { 1 } x ^ { M - 1 } \dot { + }$ $\cdot \cdot \cdot ( - 1 ) ^ { M - 1 } a _ { M - 1 } x + ( - 1 ) ^ { M } a _ { M }$ , is a homeomorphism.

Among other things, this implies that (complex) roots of a polynomial depends continuously on the coefficients. We will use this fact for our next lemma.

Finally, we establish a continuous inverse mapping for the sum-of-power function.

Lemma 6 $L e t \mathcal { X } = \{ ( x _ { 1 } , . . . , x _ { M } ) \in [ 0 , 1 ] ^ { M } : x _ { 1 } \leq x _ { 2 } \leq \cdot \cdot \cdot \leq x _ { M } \}$ . We define the sum-of-power mapping $E : \mathcal { X }  \mathcal { Z }$ by the coordinate functions

$$
Z _ { q } : = E _ { q } ( X ) : = \sum _ { m = 1 } ^ { M } ( x _ { m } ) ^ { q } , \qquad q = 0 , . . . , M .\tag{13}
$$

where $\mathcal { Z }$ is the range of the function. The function E has a continuous inverse mapping.

Proof. First of all note that ${ \mathcal { Z } } ,$ the range of $E _ { \mathrm { { : } } }$ , is a compact set. This follows from following observations:

• The domain of E is a bounded polytope (i.e. a compact set),

• E is a continuous function, and

• image of a compact set under a continuous function is a compact set.

To show the continuity of inverse mapping, we establish connection to the continuous dependence of roots of polynomials on its coefficients.

As in Lemma 4, for any $u \in \mathcal X$ , let $z = E ( u )$ and construct the polynomial:

$$
P _ { u } ( x ) = \prod _ { m = 1 } ^ { M } \left( x - u _ { m } \right)\tag{14}
$$

If we expand the polynomial we obtain:

$$
P _ { u } ( x ) = x ^ { M } - a _ { 1 } x ^ { M - 1 } + \cdot \cdot \cdot ( - 1 ) ^ { M - 1 } a _ { M - 1 } x + ( - 1 ) ^ { M } a _ { M }\tag{15}
$$

with coefficients being elementary symmetric polynomials in $u ,$ i.e.

$$
a _ { m } = \sum _ { 1 \leq j _ { 1 } < j _ { 2 } < \cdots < j _ { m } \leq M } u _ { j _ { 1 } } u _ { j _ { 2 } } \cdot \cdot \cdot u _ { j _ { m } }\tag{16}
$$

These elementary symmetric polynomials can be uniquely expressed as a function of z by Newton-Girard formula:

$$
a _ { m } = \frac { 1 } { m } \operatorname* { d e t } \left( \begin{array} { c c c c c c c } { z _ { 1 } } & { 1 } & { 0 } & { 0 } & { \cdots } & { 0 } \\ { z _ { 2 } } & { z _ { 1 } } & { 1 } & { 0 } & { \cdots } & { 0 } \\ { z _ { 3 } } & { z _ { 2 } } & { z _ { 1 } } & { 1 } & { \cdots } & { 0 } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { z _ { m - 1 } } & { z _ { m - 2 } } & { z _ { m - 3 } } & { z _ { m - 4 } } & { \cdots } & { 1 } \\ { z _ { m } } & { z _ { m - 1 } } & { z _ { m - 2 } } & { z _ { m - 3 } } & { \cdots } & { z _ { 1 } } \end{array} \right)\tag{17}
$$

Since determinants are just polynomials, a is a continuous function of $z .$ Thus to show continuity of inverse mapping of $\breve { E } ,$ , it remains to show continuity from a back to the roots u. In this regard, we invoke Theorem 5. Note that homeomorphism implies the mapping as well as its inverse is continuous. Thus, restricting to the compact set $\mathcal { Z }$ where the map from coefficients to roots only goes to the reals, the desired result follows. To explicitly check the continuity, note that limit of $E ^ { - 1 } ( z )$ as z approaches $z ^ { * }$ from inside ${ \mathcal { Z } } ,$ , always exists and is equal to $E ^ { - 1 } \dot { ( z ^ { * } ) }$ since it does so in the complex plane.

With the lemma developed above we are in a position to tackle the main theorem.

Theorem 7 Let $f : [ 0 , 1 ] ^ { M } \to \mathbb { R }$ be a permutation invariant continuous function iff it has the representation

$$
f ( x _ { 1 } , . . . , x _ { M } ) = \rho \left( \sum _ { m = 1 } ^ { M } \phi ( x _ { m } ) \right)\tag{18}
$$

for some continuous outer and innerfunction $\rho : \mathbb { R } ^ { M + 1 }  \mathbb { R }$ and $\phi : \mathbb { R } \to \mathbb { R } ^ { M + 1 }$ respectively. The inner function φ is independent of the function f.

Proof. The sufficiency follows by observing that the function $\rho \left( { \sum _ { m = 1 } ^ { M } \phi ( x _ { m } ) } \right)$ satisfies the permutation invariance condition.

To prove necessity, i.e. that all permutation invariant continuous functions over the compact set can be expressed in this manner, we divide the proof into two parts, with outline in Fig. 4. We begin

![](images/233dc68d01d2b809a6da6a357735a4add656f642f9f821a710b428fbf9d94c8f.jpg)

Figure 4: Outline of the proof strategy for Theorem 2.1. The proof consists of two parts. First, we desire to show that we can find unique embeddings for each possible input, i.e. we show that there exists a homeomorphism E of the form $\begin{array} { r } { E ( X ) = \sum _ { x \in X } \phi ( x ) } \end{array}$ between original domain and some higher dimensional space $\mathcal { Z }$ . The second part of the proof consists of showing we can map the embedding to desired target value, i.e. to show the existence of the continuous map $\rho$ between Z and original target space such that $\begin{array} { r } { f ( X ) = \rho ( \sum _ { x \in X } \phi ( x ) ) } \end{array}$ .

by looking at the continuous embedding formed by the inner function: $\begin{array} { r } { E ( X ) = \sum _ { m = 1 } ^ { M } \phi ( x _ { m } ) } \end{array}$ Consider $\phi : \mathbb { R } \to \mathbb { R } ^ { M + 1 }$ defined as $\phi ( x ) = [ 1 , x , x ^ { 2 } , . . . , x ^ { M } ]$ . Now as E is a polynomial, the image of $[ 0 , 1 ] ^ { M }$ in $\mathbb { R } ^ { M + 1 }$ under E is a compact set as well, denote it by Z. Then by definition, the embedding $\dot { E ^ { \cdot } } : [ 0 , 1 ] ^ { M } \to \mathcal { Z }$ is surjective. Using Lemma 4 and $^ { 6 , }$ we know that upon restricting the permutations, i.e. replacing $[ 0 , 1 \bar { ] } ^ { M }$ with $\mathcal { X } = \{ ( x _ { 1 } , . . . , x _ { M } ) \in [ 0 , 1 ] ^ { M } : x _ { 1 } \leq x _ { 2 } \leq \cdot \cdot \cdot \leq x _ { M } \}$ the embedding $E : \mathcal { X } \overset { \cdot } {  } \mathcal { Z }$ is injective with a continuous inverse. Therefore, combining these observation we get that E is a homeomorphism between X and Z. Now it remains to show that we can map the embedding to desired target value, i.e. to show the existence of the continuous map $\rho : \mathcal { Z } \overset { \cdot } {  } \mathbb { R }$ such that ${ \check { \rho ( E ( X ) ) } } = f ( { \check { X } } )$ ). In particular consider the map $\rho ( z ) = f ( E ^ { - 1 } ( z ) )$ . The continuity of $\rho$ follows directly from the fact that composition of continuous functions is continuous. Therefore we can always find continuous functions $\phi$ and $\rho$ to express any permutation invariant function $f$ as $\rho \left( \sum _ { m = 1 } ^ { M } \phi ( x _ { m } ) \right)$

A very similar but more general results holds in case of any continuous function (not necessarily permutation invariant). The result is known as Kolmogorov-Arnold representation theorem [47, Chap. 17] which we state below:

Theorem 8 (Kolmogorov–Arnold representation) Let $f : [ 0 , 1 ] ^ { M } \to$ R be an arbitrary multivariate continuousfunction iffit has the representation

$$
f ( x _ { 1 } , . . . , x _ { M } ) = \rho \left( \sum _ { m = 1 } ^ { M } \lambda _ { m } \phi ( x _ { m } ) \right)\tag{19}
$$

with continuous outer and innerfunctions $\rho : \mathbb { R } ^ { 2 M + 1 } \to \mathbb { R } a n d \phi : \mathbb { R } \to \mathbb { R } ^ { 2 M + 1 }$ . The inner function φ is independent ofthefunction f.

This theorem essentially states a representation theorem for any multivariate continuous function. Their representation is very similar to the one we are proved, except for the dependence of inner transformation on the co-ordinate through $\lambda _ { m }$ . Thus it is reassuring that behind all the beautiful mathematics something intuitive is happening. If the function is permutation invariant, this dependence on co-ordinate of the inner transformation gets dropped!

Further we can show that arbitrary approximator having the same form can be obtained for continuous permutation-invariant functions.

Theorem 9 Assume the elements arefrom a compact set in $\mathbb { R } ^ { d } { } _ { : }$ , i.e. possibly uncountable, and the set size isfixed to M. Then any continuousfunction operating on a set $X , i . e . f : \mathbb { R } ^ { d \times M }  \mathbb { R }$ which is permutation invariant to the elements in X can be approximated arbitrarily close in the form of $\textstyle \rho \left( \sum _ { x \in X } \phi ( x ) \right)$ ,for suitable transformations φ and $\rho .$

Proof. Permutation invariance follows from the fact that sets have no particular order, hence any function on a set must not exploit any particular order either. The sufficiency follows by observing that the function $\textstyle \rho \left( \sum _ { x \in X } \phi ( x ) \right)$ satisfies the permutation invariance condition.

To prove necessity, i.e. that all continuous functions over the compact set can be approximated arbitrarily close in this manner, we begin noting that polynomials are universal approximators by Stone–Weierstrass theorem [48, sec. 5.7]. In this case the Chevalley-Shephard-Todd (CST) theorem [49, chap. V, theorem 4], or more precisely, its special case, the Fundamental Theorem of Symmetric Functions states that symmetric polynomials are given by a polynomial of homogeneous symmetric monomials. The latter are given by the sum over monomial terms, which is all that we need since it implies that all symmetric polynomials can be written in the form required by the theorem.

Finally, we still conjecture that even in case of sets of all sizes, i.e. when the domain is $2 ^ { [ 0 , 1 ] }$ , a representation of the form $\begin{array} { r } { f ( X ) = \rho \left( \sum _ { x \in X } \phi ( x ) \right) } \end{array}$  should exist for all “continuous” permutation invariant functions for some suitable transformations $\rho$ and $\phi .$ However, in this case even what a “continuous” function means is not clear as the space $\dot { 2 } ^ { [ 0 , 1 ] }$ does not have any natural topology. As a future work, we want to study further by defining various topologies, like using Fréchet distance as used in [46] or MMD distance. Our preliminary findings in this regards hints that using MMD distance if the representation is allowed to be in $\breve { \ell } ^ { 2 }$ , instead of being finite dimensional, then the conjecture seems to be provable. Thus, clearly this direction needs further exploration. We end this section by providing some examples:

## Examples:

$x _ { 1 } x _ { 2 } ( x _ { 1 } + x _ { 2 } + 3 )$ , Consider $\phi ( x ) = [ x , x ^ { 2 } , x ^ { 3 } ]$ and $\rho ( [ u , v , w ] ) = u v - w + 3 ( u ^ { 2 } - v ) / 2$ then $\rho ( \phi ( x _ { 1 } ) + \phi ( x _ { 2 } ) )$ is the desired function.

$x _ { 1 } x _ { 2 } x _ { 3 } + x _ { 1 } + x _ { 2 } + x _ { 3 }$ , Consider $\phi ( x ) = [ x , x ^ { 2 } , x ^ { 3 } ]$ and $\rho ( [ u , v , w ] ) = ( u ^ { 3 } + 2 w -$ $3 u v ) / 6 + u ,$ then $\rho ( \phi ( x _ { 1 } ) + \phi ( x _ { 2 } ) + \phi ( x _ { 3 } ) )$ is the desired function.

$1 / n ( x _ { 1 } + x _ { 2 } + x _ { 3 } + . . . + x _ { m } )$ , Consider $\phi ( x ) = [ 1 , x ]$ and $\rho ( [ u , v ] ) = v / u .$ , then $\rho ( \phi ( x _ { 1 } ) +$ $\phi ( x _ { 2 } ) + \phi ( x _ { 3 } ) + . . . + \phi ( x _ { m } ) )$ is the desired function.

$\operatorname* { m a x } \{ x _ { 1 } , x _ { 2 } , x _ { 3 } , . . . , x _ { m } \}$ , Consider $\phi ( x ) = [ e ^ { \alpha x } , x e ^ { \alpha x } ]$ and $\rho ( [ u , v ] ) = v / u$ , then as $\alpha $ $\infty ,$ then we have $\rho ( \phi ( x _ { 1 } ) + \phi ( x _ { 2 } ) + \phi ( x _ { 3 } ) + . . . + \phi ( x _ { m } ) )$ ) approaching the desired function. • Second largest among $\{ x _ { 1 } , x _ { 2 } , x _ { 3 } , . . . , x _ { m } \}$ , Consider $\phi ( x ) = [ e ^ { \alpha x } , x e ^ { \alpha x } ]$ and $\rho ( [ u , v ] ) =$ $( v - ( v / u ) e ^ { \alpha v / u } ) / ( u - e ^ { \alpha v / u } )$ , then as $\alpha \to \infty$ , we have $\rho ( \phi ( x _ { 1 } ) + \phi ( x _ { 2 } ) + \phi ( x _ { 3 } ) + . . . +$ $\phi ( x _ { m } ) )$ approaching the desired function.

## B Proof of Lemma 3

Our goal is to design neural network layers that are equivariant to permutations of elements in the input x. The function $\mathbf { f } : \mathfrak { X } ^ { M } \to \mathcal { Y } ^ { M }$ is equivariant to the permutation of its inputs iff

$$
\mathbf { f } ( \pi \mathbf { x } ) = \pi \mathbf { f } ( \mathbf { x } ) \quad \forall \pi \in S _ { M }
$$

where the symmetric group $\boldsymbol { \mathcal { S } } _ { M }$ is the set of all permutation of indices $1 , \ldots , M .$

Consider the standard neural network layer

$$
\mathbf { f } _ { \ u { \Theta } } ( \mathbf { x } ) \doteq \pmb { \sigma } ( \ u { \Theta } \mathbf { x } ) \quad \ u { \Theta } \in \mathbb { R } ^ { M \times M }\tag{20}
$$

where $\Theta$ is the weight vector and $\sigma : \mathbb { R } $ R is a nonlinearity such as sigmoid function. The following lemma states the necessary and sufficient conditions for permutation-equivariance in this type of function.

Lemma 3 Thefunction $\mathbf { f } _ { \Theta } : \mathbb { R } ^ { M } \to \mathbb { R } ^ { M }$ as defined in (20) is permutation equivariant if and only if all the off-diagonal elements $o f \Theta$ are tied together and all the diagonal elements are equal as well. That $i s ,$

$$
\Theta = \lambda \mathbf { I } + \gamma \left( \mathbf { 1 1 } ^ { \mathsf { T } } \right) \qquad \lambda , \gamma \in \mathbb { R } \quad \mathbf { 1 } = [ 1 , \ldots , 1 ] ^ { \mathsf { T } } \in \mathbb { R } ^ { M }
$$

where $\mathbf { I } \in \mathbb { R } ^ { M \times M }$ is the identity matrix.

## Proof.

From definition of permutation equivariance $\mathbf { f } _ { \Theta } ( \pi \mathbf { x } ) = \pi \mathbf { f } _ { \Theta } ( \mathbf { x } )$ and definition of f in $( 2 0 ) .$ , the condition becomes $\bar { \pmb { \sigma } } ( \Theta \pi \mathbf { x } ) = \pi \pmb { \sigma } ( \Theta \mathbf { x } )$ , which (assuming sigmoid is a bijection) is equivalent to $\Theta \pi = \pi \Theta$ . Therefore we need to show that the necessary and sufficient conditions for the matrix $\Theta \in \mathbb { R } ^ { M \times M }$ to commute with all permutation matrices $\pi \in { \cal S } _ { M }$ is given by this proposition. We prove this in both directions:

• To see why $\Theta = \lambda \mathbf { I } + \gamma \mathbf { \Theta } ( \mathbf { 1 1 } ^ { \mathsf { T } } )$ commutes with any permutation matrix, first note that commutativity is linear – that is

$$
\Theta _ { 1 } \pi = \pi \Theta _ { 1 } \wedge \Theta _ { 2 } \pi = \pi \Theta _ { 2 } \quad \Rightarrow \quad ( a \Theta _ { 1 } + b \Theta _ { 2 } ) \pi = \pi ( a \Theta _ { 1 } + b \Theta _ { 2 } ) .
$$

Since both Identity matrix I, and constant matrix $\mathbf { 1 1 ^ { \mathsf { T } } }$ , commute with any permutation matrix, so does their linear combination $\Theta = \lambda \mathbf { I } + \gamma \mathbf { \Gamma } ( \mathbf { 1 1 } ^ { \mathsf { T } } )$

• We need to show that in a matrix Θ that commutes with $ { \mathbf { \hat { \mu } } } ^ { 6 6 }  { \mathrm { a l l } } ^ { 5 } $ permutation matrices

– All diagonal elements are identical: Let $\pi _ { k , l }$ for $1 \leq k , l \leq M , k \neq l$ , be a transposition (i.e. a permutation that only swaps two elements). The inverse permutation matrix of $\pi _ { k , l }$ is the permutation matrix of $\pi _ { l , k } = \pi _ { k , l } ^ { \mathsf { T } }$ . We see that commutativity of Θ with the transposition $\pi _ { k , l }$ implies that $\Theta _ { k , k } = \Theta _ { l , l } \vdots$

$$
\pi _ { k , l } \Theta = \Theta \pi _ { k , l } \Rightarrow \pi _ { k , l } \Theta \pi _ { l , k } = \Theta \Rightarrow ( \pi _ { k , l } \Theta \pi _ { l , k } ) _ { l , l } = \Theta _ { l , l } \Rightarrow \Theta _ { k , k } = \Theta _ { l , l }
$$

Therefore, π and $\Theta$ commute for any permutation $\pi ,$ they also commute for any transposition $\pi _ { k , l }$ and therefore $\Theta _ { i , i } = \dot { \lambda } \dot { \forall i }$

– All off-diagonal elements are identical: We show that since Θ commutes with any product of transpositions, any choice two off-diagonal elements should be identical. $\operatorname { L e t } \left( i , j \right)$ and $( i ^ { \prime } , j ^ { \prime } )$ be the index of two off-diagonal elements $( i . e . \ i \ne j$ and $i ^ { \prime } \neq j ^ { \prime } )$ Moreover for now assume $i \neq i ^ { \prime }$ and $j \neq j ^ { \prime }$ . Application of the transposition $\pi _ { i , i ^ { \prime } } \Theta$ swaps the rows $i , i ^ { \prime }$ in Θ. Similarly, $\Theta \pi _ { j , j ^ { \prime } }$ switches the $j ^ { t h }$ column with $j ^ { \prime t h }$ column. From commutativity property of Θ and $\ddot { \pi } \in S _ { n }$ we have

$$
\pi _ { j ^ { \prime } , j } \pi _ { i , i ^ { \prime } } \Theta = \Theta \pi _ { j ^ { \prime } , j } \pi _ { i , i ^ { \prime } } \Rightarrow \pi _ { j ^ { \prime } , j } \pi _ { i , i ^ { \prime } } \Theta ( \pi _ { j ^ { \prime } , j } \pi _ { i , i ^ { \prime } } ) ^ { - 1 } = \Theta
$$

$$
\begin{array} { r } { \pi _ { j ^ { \prime } , j } \pi _ { i , i ^ { \prime } } \Theta \pi _ { i ^ { \prime } , i } \pi _ { j , j ^ { \prime } } = \Theta \Rightarrow ( \pi _ { j ^ { \prime } , j } \pi _ { i , i ^ { \prime } } \Theta \pi _ { i ^ { \prime } , i } \pi _ { j , j ^ { \prime } } ) _ { i , j } = \Theta _ { i , j } \quad \Rightarrow \Theta _ { i ^ { \prime } , j ^ { \prime } } = \Theta _ { i , j ^ { \prime } } } \end{array}
$$

where in the last step we used our assumptions that $i \neq i ^ { \prime } , j \neq j ^ { \prime } , i \neq j$ and $i ^ { \prime } \neq j ^ { \prime }$ . In the cases where either $i = i ^ { \prime } \operatorname { o r } j = j ^ { \prime }$ , we can use the above to show that $\Theta _ { i , j } = \Theta _ { i ^ { \prime \prime } , j ^ { \prime \prime } }$ and $\Theta _ { i ^ { \prime } , j ^ { \prime } } = \Theta _ { i ^ { \prime \prime } , j ^ { \prime \prime } }$ 0 , for some $\bar { i } ^ { \prime \prime } \ne \ i , i ^ { \prime }$ and $\boldsymbol { j } ^ { \prime \prime } \neq \boldsymbol { j } , \boldsymbol { j } ^ { \prime }$ , and conclude $\Theta _ { i , j } \stackrel {  } { = } \Theta _ { i ^ { \prime } , j ^ { \prime } }$

## C More Details on the architecture

## C.1 Invariant model

The structure of permutation invariant functions in Theorem 2 hints at a general strategy for inference over sets of objects, which we call deep sets. Replacing φ and $\rho$ by universal approximators leaves matters unchanged, since, in particular, φ and ρ can be used to approximate arbitrary polynomials. Then, it remains to learn these approximators. This yields in the following model:

• Each instance $x _ { m } \forall 1 \ \leq \ m \leq \ M$ is transformed (possibly by several layers) into some representation $\phi ( x _ { m } )$

• The addition $\textstyle \sum _ { m } ^ { * } \phi ( x _ { m } )$ of these representations processed using the $\rho$ network very much in the same manner as in any deep network $( e . g .$ fully connected layers, nonlinearities, etc).

• Optionally: If we have additional metainformation $z ,$ then the above mentioned networks could be conditioned to obtain the conditioning mapping $\phi ( x _ { m } | z )$

![](images/76c8076cbdf7a97ffd9e2d12234dccee0aa69f5e4f6094a39ae4b691286da39f.jpg)  
Figure 5: Architecture of DeepSets: Invariant

In other words, the key to deep sets is to add up

all representations and then apply nonlinear transformations.

The overall model structure is illustrated in Fig. 7.

This architecture has a number of desirable properties in terms of universality and correctness. We assume in the following that the networks we choose are, in principle, universal approximators. That is, we assume that they can represent any functional mapping. This is a well established property (see e.g. [50] for details in the case of radial basis function networks).

What remains is to state the derivatives with regard to this novel type of layer. Assume parametrizations $w _ { \rho }$ and $w _ { \phi }$ for $\rho$ and φ respectively. Then we have

$$
\partial _ { w _ { \phi } } \rho \left( \sum _ { x ^ { \prime } \in X } \phi ( x ^ { \prime } ) \right) = \rho ^ { \prime } \left( \sum _ { x ^ { \prime } \in X } \phi ( x ) \right) \sum _ { x ^ { \prime } \in X } \partial _ { w _ { \phi } } \phi ( x ^ { \prime } )
$$

This result reinforces the common knowledge of parameter tying in deep networks when ordering is irrelevant. Our result backs this practice with theory and strengthens it by proving that it is the only way to do it.

## C.2 Equivariant model

Consider the standard neural network layer

$$
f _ { \Theta } ( \mathbf { x } ) = \sigma ( \Theta \mathbf { x } )\tag{21}
$$

where $\Theta \in \mathbb { R } ^ { M \times M }$ is the weight vector and $\sigma : \mathbb { R } ^ { M } \to \mathbb { R } ^ { M }$ is a point-wise nonlinearity such as a sigmoid function. The following lemma states the necessary and sufficient conditions for permutationequivariance in this type of function.

Lemma 3 The function $f _ { \Theta } ( \mathbf { x } ) ~ = ~ \sigma ( \Theta \mathbf { x } ) ~ f o r ~ \Theta ~ \in ~ \mathbb { R } ^ { M \times M } ~ i s$ permutation equivariant, iff all the off-diagonal elements of Θ are tied together and all the diagonal elements are equal as well. That $i s ,$

$$
\Theta = \lambda \mathbf { I } + \gamma \left( \mathbf { 1 1 } ^ { \mathsf { T } } \right) \qquad \lambda , \gamma \in \mathbb { R } \quad \mathbf { 1 } = [ 1 , \ldots , 1 ] ^ { \mathsf { T } } \in \mathbb { R } ^ { M }
$$

where $\mathbf { I } \in \mathbb { R } ^ { M \times M }$ is the identity matrix.

This function is simply a non-linearity applied to a weighted combination of i) its input Ix and; ii) the sum of input values $( \mathbf { 1 1 } ^ { \mathsf { T } } ) \mathbf { x }$ Since summation does not depend on the permutation, the layer is permutation-equivariant. Therefore we can manipulate the operations and parameters in this layer, for example to get another variation $f ( \mathbf { x } ) = \overbar { \boldsymbol { \sigma } } ( \lambda \mathbf { I x } + \gamma \overbar { \mathrm { m a x p o o l } } ( \mathbf { x } ) \mathbf { 1 } )$ , where the maxpooling operation over elements of the set (similarly to summation) is commutative. In practice using this variation performs better in some applications.

![](images/dbfc120f4eeff7042865ddc62e94f3bc7b9b094566a62251b1135aaeaff56124.jpg)  
Figure 6: Illustration of permutation equivariant layer. Same color indicates weight sharing.

So far we assumed that each instance $x _ { m } \in \mathbb { R }$ $\mathrm { ~ - ~ } i . e .$ a single input and also output channel. For multiple input-output channels, we may speed up the operation of the layer using matrix multiplication. For $D / D ^ { \prime }$ input/output channels $( i . e . \textbf { x } \in ~ \mathbb { R } ^ { M \times D } , \textbf { y } \in ~ \mathbb { R } ^ { M \times D ^ { \prime } }$ , this layer becomes

![](images/47563c7b9d3df25d7d5adcf7179ed9dbc42c4c36912cf75e39649882a2155960.jpg)

$$
f ( \mathbf { x } ) = \sigma \big ( \mathbf { x } \Lambda - \mathbf { 1 1 } ^ { \mathsf { T } } \mathbf { x } \Gamma \big )\tag{22}
$$

Figure 7: Architecture of DeepSets: Equivariant

where $\boldsymbol { \Lambda } , \boldsymbol { \Gamma } \in \mathbb { R } ^ { D \times D ^ { \prime } }$ are model parameters. As

before, we can have a maxpool version as: $f ( \mathbf { x } ) = \sigma ( \mathbf { x } \Lambda - \mathbf { 1 } \mathrm { m a x p o o l } ( \mathbf { x } ) \Gamma )$ where maxpoo $| ( \mathbf { x } ) =$ $( \operatorname* { m a x } _ { m } \mathbf { x } ) \in \mathbb { R } ^ { 1 \times D }$ is a row-vector of maximum value of x over the $\mathbf { \bar { s e t } } ^ { \prime \prime }$ dimension. We may further reduce the number of parameters in favor of better generalization by factoring Γ and Λ and keeping a single $\boldsymbol { \Lambda } \in \mathbb { R } ^ { D , D ^ { \prime } }$ and $\beta \in \mathbb { R } ^ { D ^ { \prime } }$

$$
f ( \mathbf { x } ) = \sigma \big ( \beta + ( \mathbf { x } - \mathbf { 1 } \mathrm { m a x p o o l } ( \mathbf { x } ) \big ) \Gamma \big )\tag{23}
$$

Stacking: Since composition of permutation equivariant functions is also permutation equivariant, we can build deep models by stacking layers of (23). Moreover, application of any commutative pooling operation (e.g. max-pooling) over the set instances produces a permutation invariant function.

![](images/c89c9c03b9040f30f8f56ea3d8444646e015ec672cf01c30f4227cd838b61831.jpg)  
Figure 8: Using multiple permutation equivariant layers. Since permutation equivariance compose we can stack multiple such layers

## D Bayes Set [36]

Bayesian sets consider the problem of estimating the likelihood of subsets X of a ground set X . In general this is achieved by an exchangeable model motivated by deFinetti’s theorem concerning exchangeable distributions via

$$
p ( X | \alpha ) = \int d \theta \left[ \prod _ { m = 1 } ^ { M } p ( x _ { m } | \theta ) \right] p ( \theta | \alpha ) .\tag{24}
$$

This allows one to perform set expansion, simply via the score

$$
s ( x | X ) = \log \frac { p ( X \cup \{ x \} | \alpha ) } { p ( X | \alpha ) p ( \{ x \} | \alpha ) }\tag{25}
$$

Note that $s ( x | X )$ is the pointwise mutual information between x and X. Moreover, due to exchangeability, it follows that regardless of the order of elements we have

$$
S ( X ) : = \sum _ { m = 1 } ^ { M } s \left( x _ { m } \vert \left\{ x _ { m - 1 } , \ldots x _ { 1 } \right\} \right) = \log p ( X | \alpha ) - \sum _ { m = 1 } ^ { M } \log p ( \left\{ x _ { m } \right\} | \alpha )\tag{26}
$$

In other words, we have a set function log $p ( X | \alpha )$ with a modular term-dependent correction. When inferring sets it is our goal to find set completions $\{ x _ { m + 1 } , \hdots x _ { M } \}$ for an initial set of query terms $\{ x _ { 1 } , \ldots , x _ { m } \}$ such that the aggregate set is well coherent. This is the key idea of the Bayesian Set algorithm.

## D.1 Exponential Family

In exponential families, the above approach assumes a particularly nice form whenever we have conjugate priors. Here we have

$$
p ( x | \theta ) = \exp \left( \langle \phi ( x ) , \theta \rangle - g ( \theta ) \right) { \mathrm { ~ a n d ~ } } p ( \theta | \alpha , M _ { 0 } ) = \exp \left( \langle \theta , \alpha \rangle - M _ { 0 } g ( \theta ) - h ( \alpha , M _ { 0 } ) \right) .\tag{27}
$$

The mapping $\phi : x  { \mathcal { F } }$ is usually referred as sufficient statistic of x which maps x into a feature space $\bar { \mathcal { F } }$ . Moreover, $g ( \theta )$ is the log-partition (or cumulant-generating) function. Finally, $p ( \theta | \alpha , M _ { 0 } )$ denotes the conjugate distribution which is in itself a member of the exponential family. It has the normalization $\begin{array} { r } { \bar { h } ( \bar { \alpha } , M _ { 0 } ) = \int d \theta \exp \left( \langle \theta , \alpha \rangle - M _ { 0 } g ( \theta ) \right) } \end{array}$ ). The advantage of this is that $\bar { s } ( x | X )$ and $S ( X )$ can be computed in closed form [36] via

$$
s ( X ) = h \left( \alpha + \phi ( X ) , M _ { 0 } + M \right) + ( M - 1 ) h ( \alpha , M _ { 0 } ) - \sum _ { m = 1 } ^ { M } h ( \alpha + \phi ( x _ { m } ) , M + 1 )\tag{28}
$$

$$
\begin{array} { r } { s ( x | X ) = h ( \alpha + \phi ( \{ x \} \cup X ) , M _ { 0 } + M + 1 ) + h ( \alpha , M _ { 0 } ) } \\ { - h ( \alpha + \phi ( X ) , M _ { 0 } + M ) - h ( \alpha + \phi ( x ) , M + 1 ) } \end{array}\tag{29}
$$

For convenience we defined the sufficient statistic of a set to be the sum over its constituents, i.e. $\begin{array} { r } { \phi ( X ) = \sum _ { m } \phi ( x _ { m } ) } \end{array}$ . It allows for very simple computation and maximization over additional elements to be added to X, since $\phi ( X )$ can be precomputed.

## D.2 Beta-Binomial Model

The model is particularly simple when dealing with the Binomial distribution and its conjugate Beta prior, since the ratio of Gamma functions allows for simple expressions. In particular, we have

$$
h ( \beta ) = \log \Gamma ( \beta ^ { + } ) + \log \Gamma ( \beta ^ { - } ) - \Gamma ( \beta ) .\tag{30}
$$

With some slight abuse of notation we let $\alpha = ( \beta ^ { + } , \beta ^ { - } )$ and $M _ { 0 } = \beta ^ { + } + \beta ^ { - }$ . Setting $\phi ( 1 ) = ( 1 , 0 )$ and ${ \phi ( 0 ) = ( 0 , 1 ) }$ allows us to obtain $\phi ( X ) = ( M ^ { + } , M ^ { - } ) , \mathrm { i . e . } \phi ( X )$ contains the counts of occurrences of $x _ { m } = 1$ and $x _ { m } = 0$ respectively. This leads to the following score functions

$$
\begin{array} { l } { { s ( X ) = \log \Gamma ( \beta ^ { + } + M ^ { + } ) + \log \Gamma ( \beta ^ { - } + M ^ { - } ) - \log \Gamma ( \beta + M ) } } \\ { { \mathrm { ~ } } } \\ { { \displaystyle ~ - \log \Gamma ( \beta ^ { + } ) - \log \Gamma ( \beta ^ { - } ) + \log \Gamma ( \beta ) - M ^ { + } \log \frac { \beta ^ { + } } { \beta } - M ^ { - } \log \frac { \beta ^ { - } } { \beta } } } \end{array}\tag{31}
$$

$$
s ( x | X ) = { \left\{ \begin{array} { l l } { \log { \frac { \beta ^ { + } + M ^ { + } } { \beta + M } } - \log { \frac { \beta ^ { + } } { \beta } } { \mathrm { ~ i f ~ } } x = 1 } \\ { \log { \frac { \beta ^ { - } + M ^ { - } } { \beta + M } } - \log { \frac { \beta ^ { - } } { \beta } } { \mathrm { ~ o t h e r w i s e } } } \end{array} \right. }\tag{32}
$$

This is the model used by [36] when estimating Bayesian Sets for objects. In particular, they assume that for any given object x the vector $\phi ( x ) \in \{ 0 ; 1 \} ^ { d }$ is a d-dimensional binary vector, where each coordinate is drawn independently from some Beta-Binomial model. The advantage of the approach is that it can be computed very efficiently while only maintaining minimal statistics of X.

In a nutshell, the algorithmic operations performed in the Beta-Binomial model are as follows:

$$
s ( x | X ) = 1 ^ { \top } \left[ \sigma \left( \sum _ { m = 1 } ^ { M } \phi ( x _ { m } ) + \phi ( x ) + \beta \right) - \sigma \left( \phi ( x ) + \beta \right) \right]\tag{33}
$$

In other words, we sum over statistics of the candidates $x _ { m } ,$ , add a bias term $\beta ,$ perform a coordinatewise nonlinear transform over the aggregate statistic (in our case a logarithm), and finally we aggregate over the so-obtained scores, weighing each contribution equally. $s { \bar { ( } } X )$ is expressed analogously.

## D.3 Gauss Inverse Wishart Model

Before abstracting away the probabilistic properties of the model, it is worth paying some attention to the case where we assume that $x _ { i } \sim \mathcal { N } ( \bar { \mu } , \bar { \Sigma } )$ and $( \mu , \Sigma ) \sim \mathrm { N I W } ( \mu _ { 0 } , \lambda , \bar { \Psi _ { \star } } \bar { \nu _ { \lambda } }$ , for a suitable set of conjugate parameters. While the details are (arguably) tedious, the overall structure of the model is instructive.

First note that the sufficient statistic of the data $x \in \mathbb { R } ^ { d }$ is now given by $\phi ( x ) = ( x , x x ^ { \top } )$ . Secondly, note that the conjugate log-partition function h amounts to computing determinants of terms involving $\textstyle \sum _ { m } x _ { m } x _ { m } ^ { \top }$ and moreover, nonlinear combinations of the latter with $\sum _ { m } x _ { m }$

The algorithmic operations performed in the Gauss Inverse Wishart model are as follows:

$$
s ( x | X ) = \sigma \left( \sum _ { m = 1 } ^ { M } \phi ( x _ { m } ) + \phi ( x ) + \beta \right) - \sigma \left( \phi ( x ) + \beta \right)\tag{34}
$$

Here $\sigma$ is a nontrivial convex function acting on a (matrix, vector) pair and $\phi ( x )$ is no longer a trivial map but performs a nonlinear dimension altering transformation on x. We will use this general template to fashion the Deep Sets algorithm.

## E Text Concept Set Retrieval

We consider the task of text concept set retrieval, where the objective is to retrieve words belonging to a ‘concept’ or ‘cluster’, given few words from that particular concept. For example, given the set of words {tiger, lion, cheetah}, we would need to retrieve other related words like jaguar, puma, etc, which belong to the same concept of big cats. The model implicitly needs to reason out the concept connecting the given set and then retrieve words based on their relevance to the inferred concept. Concept set retrieval is an important due to wide range of potential applications including personalized information retrieval, tagging large amounts of unlabeled or weakly labeled datasets, etc. This task of concept set retrieval can be seen as a set completion task conditioned on the latent semantic concept, and therefore our DeepSets form a desirable approach.

Dataset To construct a large dataset containing sets of related words, we make use of Wikipedia text due to its huge vocabulary and concept coverage. First, we run topic modeling on publicly available wikipedia text with $\check { K }$ number of topics. Specifically, we use the famous latent Dirichlet allocation [38, 39], taken out-of-the-box<sup>4</sup>. Next, we choose top $N _ { T } = 5 0$ words for each latent topic as a set giving a total of $K$ sets of size $N _ { T }$ . To compare across scales, we consider three values of $k = \{ \tilde { 1 } k , 3 \tilde { k } , 5 k \}$ } giving us three datasets LDA-1k, LDA-3k, and LDA-5k, with corresponding vocabulary sizes of 17k, 38k, and 61k. Few of the topics from LDA-1k are visualized in Tab. 9.

Methods Our DeepSets model uses a feedforward neural network (NN) to represent a query and each element of a set, i.e., φ(x) for an element x is encoded as a NN. Specifically, $\phi ( x )$ represents each word via 50-dimensional embeddings that are we learn jointly, followed by two fully connected layers of size 150, with ReLU activations. We then construct a set representation or feature, by sum pooling all the individual representations of its elements, along with that of the query. Note that this sum pooling achieves permutation invariance, a crucial property of our DeepSets (Theorem 2). Next, use input this set feature into another NN to assign a single score to the set, shown as $\rho ( . ) .$ We instantiate $\rho ( . )$ as three fully connected layers of sizes {150, 75, 1} with ReLU activations. In summary, our DeepSets consists of two neural networks – (a) to extract representations for each element, and (b) to score a set after pooling representations of its elements.

Baselines We compare to several baselines: (a) Random picks a word from the vocabulary uniformly at random. (b) Bayes Set [36], and (c) w2v-Near that computes the nearest neighbors in the word2vec [40] space. Note that both Bayes Set and w2v NN are strong baselines. The former runs Bayesian inference using Beta-Binomial conjugate pair, while the latter uses the powerful 300 dimensional word2vec trained on the billion word GoogleNews corpus<sup>5</sup>. (d) NN-max uses a similar architecture as our DeepSets with an important difference. It uses max pooling to compute the set feature, as opposed to DeepSets which uses sum pooling. (e) NN-max-con uses max pooling on set elements but concatenates this pooled representation with that of query for a final set feature. (f) NN-sum-con is similar to NN-max-con but uses sum pooling followed by concatenation with query representation.

Evaluation To quantitatively evaluate, we consider the standard retrieval metrics – recall@K, median rank and mean reciprocal rank. To elaborate, recall@K measures the number of true labels that were recovered in the top K retrieved words. We use three values of $\mathbf { K } = \{ 1 0 , 1 0 0 , 1 k \}$ . The other two metrics, as the names suggest, are the median and mean of reciprocals of the true label ranks, respectively. Each dataset is split into TRAIN (80%), VAL (10%) and TEST (10%). We learn models using TRAIN and evaluate on TEST, while VAL is used for hyperparameter selection and early stopping.

Results and Observations Tab. 3 contains the results for the text concept set retrieval on LDA-1k, LDA-3k, and LDA-5k datasets. We summarize our findings below: (a) Our /deepsets model outperforms all other approaches on LDA-3k and LDA-5k by any metric, highlighting the significance of permutation invariance property. For instance, /deepsets is better than the w2v-Near baseline by 1.5% in Recall@10 on LDA-5k. (b) On LDA-1k, neural network based models do not perform well when compared to w2v-Near. We hypothesize that this is due to small size of the dataset insufficient to train a high capacity neural network, while w2v-Near has been trained on a billion word corpus. Nevertheless, our approach comes the closest to w2v-Near amongst other approaches, and is only 0.5% lower by Recall@10.

<table><tr><td rowspan=1 colspan=1>Topic 1</td><td rowspan=1 colspan=1>Topic 2</td><td rowspan=1 colspan=1>Topic 3</td><td rowspan=1 colspan=1>Topic 4</td><td rowspan=1 colspan=1>Topic 5</td><td rowspan=1 colspan=1>Topic 6</td></tr><tr><td rowspan=1 colspan=2>legend       president</td><td rowspan=1 colspan=1>plan</td><td rowspan=1 colspan=1>newspaper</td><td rowspan=1 colspan=1>round</td><td rowspan=1 colspan=1>point</td></tr><tr><td rowspan=1 colspan=1>airy</td><td rowspan=1 colspan=1>vice</td><td rowspan=1 colspan=1>proposed</td><td rowspan=1 colspan=1>daily</td><td rowspan=1 colspan=1>teams</td><td rowspan=1 colspan=1>angle</td></tr><tr><td rowspan=1 colspan=1>tale</td><td rowspan=1 colspan=1>served</td><td rowspan=1 colspan=1>plans</td><td rowspan=1 colspan=1>paper</td><td rowspan=1 colspan=1>final</td><td rowspan=1 colspan=1>axis</td></tr><tr><td rowspan=1 colspan=2>witch          office</td><td rowspan=1 colspan=1>proposal</td><td rowspan=1 colspan=1>news</td><td rowspan=1 colspan=1>played</td><td rowspan=1 colspan=1>plane</td></tr><tr><td rowspan=1 colspan=2>devil         elected</td><td rowspan=1 colspan=1>planning</td><td rowspan=1 colspan=1>press</td><td rowspan=1 colspan=2>redirect       direction</td></tr><tr><td rowspan=1 colspan=2>giant        secretary</td><td rowspan=1 colspan=1>approved</td><td rowspan=1 colspan=1>published</td><td rowspan=1 colspan=2>won         distance</td></tr><tr><td rowspan=1 colspan=2>story       presidency</td><td rowspan=1 colspan=1>planned</td><td rowspan=1 colspan=1>newspapers</td><td rowspan=1 colspan=1>competition</td><td rowspan=1 colspan=1>surface</td></tr><tr><td rowspan=1 colspan=2>folklore     presidential</td><td rowspan=1 colspan=1>development</td><td rowspan=1 colspan=1>editor</td><td rowspan=1 colspan=2>tournament      curve</td></tr></table>

Figure 9: Examples from our LDA-1k datasets. Notice that each of these are latent topics of LDA and hence are semantically similar.

## F Image Tagging

We next experiment with image tagging, where the task is to retrieve all relevant tags corresponding to an image. Images usually have only a subset of relevant tags, therefore predicting other tags can help enrich information that can further be leveraged in a downstream supervised task. In our setup, we learn to predict tags by conditioning /deepsets on the image. Specifically, we train by learning to predict a partial set of tags from the image and remaining tags. At test time, we the test image is used to predict relevant tags.

Datasets We report results on the following three datasets:

(a) ESPgame [51]: Contains around 20k images spanning logos, drawings, and personal photos, collected interactively as part of a game. There are a total of 268 unique tags, with each image having 4.6 tags on average and a maximum of 15 tags.

(b) IAPRTC-12.5 [52]: Comprises of around 20k images including pictures of different sports and actions, photographs of people, animals, cities, landscapes, and many other aspects of contemporary life. A total of 291 unique tags have been extracted from captions for the images. For the above two datasets, train/test splits are similar to those used in previous works [41, 44].

(c) COCO-Tag: We also construct a dataset in-house, based on MSCOCO dataset[53]. COCO is a large image dataset containing around 80k train and 40k test images, along with five caption annotations. We extract tags by first running a standard spell checker<sup>6</sup> and lemmatizing these captions. Stopwords and numbers are removed from the set of extracted tags. Each image has 15.9 tags on an average and a maximum of 46 tags. We show examples of image tags from COCO-Tag in Fig. 10. The advantages of using COCO-Tag are three fold–richer concepts, larger vocabulary and more tags per image, making this an ideal dataset to learn image tagging using /deepsets.

Image and Word Embeddings Our models use features extracted from Resnet, which is the state-of-the-art convolutional neural network (CNN) on ImageNet 1000 categories dataset using the publicly available 152-layer pretrained model<sup>7</sup>. To represent words, we jointly learn embeddings with the rest of /deepsets neural network for ESPgame and IAPRTC-12.5 datasets. But for COCO-Tag, we bootstrap from 300 dimensional word2vec embeddings<sup>8</sup> as the vocabulary for COCO-Tag is significantly larger than both ESPgame and IAPRTC-12.5 (13k vs 0.3k).

Methods The setup for DeepSets to tag images is similar to that described in Appendix E. The only difference being the conditioning on the image features, which is concatenated with the set feature obtained from pooling individual element representations. In particular, φ(x) represents each word via 300-dimensional word2vec embeddings, followed by two fully connected layers of size 300, with ReLU activations, to construct the set representation or features. As mentioned earlier, we concatenate the image features and pass this set features into another NN to assign a single score to the set, shown as ρ(.). We instantiate ρ(.) as three fully connected layers of sizes {300, 150, 1} with ReLU activations. The resulting feature forms the new input to a neural network used to score the set, in this case, score the relevance of a tag to the image.

Baselines We perform comparisons against several baselines, previously reported from [41]. Specifically, we have Least Sq., a ridge regression model, MBRM [42], JEC [43] and FastTag [41]. Note that these methods do not use deep features for images, which could lead to an unfair comparison. As there is no publicly available code for MBRM and JEC, we cannot get performances of these models with Resnet extracted features. However, we report results with deep features for FastTag and Least Sq., using code made available by the authors <sup>9</sup>.

Pred clock tower sky building tall large cloudy front city

GT traffic city building tall large tower European front clock

GT Pred laptop refrigerator person fridge screen room room magnet desk cabinet living kitchen counter shelf computer wall monitor counter

Evaluation For ESPgame and IAPRTC-12.5, we follow the evaluation metrics as in [44] – precision (P), recall (R), F1 score (F1) and number of tags with non-zero recall (N+). Note that these metrics are evaluate for each tag and the mean is reported. We refer to [44] for further details. For COCO-Tag, however, we use recall@K for three values of K = {10, 100, 1000}, along with median rank and mean reciprocal rank (see evaluation in Appendix E for metric details).

Results and Observations Tab. 4 contains the results of image tagging on ESPgame and IAPRTC-12.5, and Tab. 5 on COCO-Tag. Here are the key observations from Tab. 4: (a) The performance of /deepsets is comparable to the best of other approaches on all metrics but precision. (b) Our recall beats the best approach by 2% in ESPgame. On further investigation, we found that /deepsets retrieves more relevant tags, which are not present in list of ground truth tags due to a limited 5 tag annotation. Thus, this takes a toll on precision while gaining on recall, yet yielding improvement in F1. On the larger and richer COCO-Tag, we see that /deepsets approach outperforms other methods comprehensively, as expected. We show qualitative examples in Fig. 10.

![](images/be65b254c388020f0ec7579e3498401723eb4a2322d904e64f4b14c0430bfb2b.jpg)

![](images/a74ac622524fc24d95ccf5958512fdf39e58e3de477e539b7d30bef3612096b7.jpg)

![](images/1fa089e565c3a371b70d772803dea48b654361da59680a04881d6c3aab01e591.jpg)  
Pred person group man table sit room woman couple gather

![](images/a639a9bc3263ec105ecd76c1e963ddac8c9f54f6f147428336b7c6ac3e00b1f4.jpg)

GT Pred   
photograph ski   
snowboarder snow snow slope glide person hill snowy show hill person man slope skiing young skier

![](images/8383b1646992b840068d0467e4402fa7a952f7111eaa28347a43eb0dbe4c9940.jpg)

![](images/a86ccba0f7406326b76d2bfa3a6c6879817eee5aa57f8b766c49fd6b8d20be8f.jpg)  
Figure 10: Qualitative examples of image tagging using /deepsets. Top row: Positive examples where most of the retrieved tags are present in the ground truth (brown) or are relevant but not present in the ground truth (green). Bottom row: Few failure cases with irrelevant/wrong tags (red). From left to right, (i) Confusion between snowboarding and skiing, (ii) Confusion between back of laptop and refrigerator due to which other tags are kitchen-related, (iii) Hallucination of airplane due to similar shape of surfboard.

## G Improved Red-shift Estimation Using Clustering Information

An important regression problem in cosmology is to estimate the red-shift of galaxies, corresponding to their age as well as their distance from us [33]. Two common types of observation for distant galaxies include a) photometric and b) spectroscopic observations, where the latter can produce more accurate red-shift estimates.

One way to estimate the red-shift from photometric observations is using a regression model [34]. We use a multi-layer Perceptron for this purpose and use the more accurate spectroscopic red-shift estimates as the ground-truth. As another baseline, we have a photometric redshift estimate that is provided by the catalogue and uses various observations (including clustering information) to estimate individual galaxy-red-shift. Our objective is to use clustering information of the galaxies to improve our red-shift prediction using the multi-layer Preceptron.

Note that the prediction for each galaxy does not change by permuting the members of the galaxy cluster. Therefore, we can treat each galaxy cluster as a “set” and use permutation-equivariant layer to estimate the individual galaxy red-shifts.

For each galaxy, we have 17 photometric features <sup>10</sup> from the redMaPPer galaxy cluster catalog [35], which contains photometric readings for 26,111 red galaxy clusters. In this task in contrast to the previous ones, sets have different cardinalities; each galaxy-cluster in this catalog has between ∼ 20−300 galaxies – ${ \bf \nabla } \cdot i . e . { \bf x } \in \mathbb { R } ^ { N ( c ) \times 1 7 }$ , where $N ( c )$ is the cluster-size. See Fig. 11(a) for distribution of cluster sizes. The catalog also provides accurate spectroscopic red-shift estimates for a subset of these galaxies as well as photometric estimates that uses clustering information. Fig. 11(b) reports the distribution of available spectroscopic red-shift estimates per cluster.

We randomly split the data into 90% training and 10% test clusters, and use the following simple architecture for semi-supervised learning. We use four permutation-equivariant layers with 128, 128, 128 and 1 output channels respectively, where the output of the last layer is used as red-shift estimate. The squared loss of the prediction for available spectroscopic red-shifts is minimized.<sup>11</sup> Fig. 11(c) shows the agreement of our estimates with spectroscopic readings on the galaxies in the test-set with spectroscopic readings. The figure also compares the photometric estimates provided by the catalogue [35], to the ground-truth. As it is customary in cosmology literature, we report the average scatter $\frac { | z _ { \mathrm { s p e c } } - z | } { 1 + z _ { \mathrm { s p e c } } }$ , where $z _ { \mathrm { s p e c } }$ is the accurate spectroscopic measurement and z is a photometric estimate. The average scatter using our model is .023 compared to the scatter of .025 in the original photometric estimates for the redMaPPer catalog. Both of these values are averaged over all the galaxies with spectroscopic measurements in the test-set.

We repeat this experiment, replacing the permutation-equivariant layers with fully connected layers (with the same number of parameters) and only use the individual galaxies with available spectroscopic estimate for training. The resulting average scatter for multi-layer Perceptron is .026, demonstrating that using clustering information indeed improves photometric red-shift estimates.

![](images/a9aa0601c480bb3c6d5eab37f7f3b61c51d30f175a5aa2b91667605390db9a3c.jpg)

![](images/b8745320816523e3d819d74daa453809b3c85cb990629d97919434b0f19b792b.jpg)

![](images/15282fbe1e4fda6729b59fb230262ac0c7a2a7dafa3019132d843d936bab0e7d.jpg)  
Figure 11: Application of permutation-equivariant layer to semi-supervised red-shift prediction using clustering information: (a) distribution of cluster (set) size; (b) distribution of reliable red-shift estimates per cluster; (c) prediction of red-shift on test-set (versus ground-truth) using clustering information as well as RedMaPPer photometric estimates (also using clustering information).

![](images/ffa24c84a5213c1ec57e392949cffbc8a541e5688b91f549617d3755d2b894b1.jpg)  
Figure 12: Examples for 8 out of 40 object classes (column) in the ModelNet40. Each point-cloud is produces by sampling 1000 particles from the mesh representation of the original MeodelNet40 instances. Two point-clouds in the same column are from the same class. The projection of particles into xy, zy and xz planes are added for better visualization.

## H Point Cloud Classification

Tab. 6 presents a more detailed result on classification performance, using different techniques. Fig. 12 shows examples of the dataset used for training. Fig. 13 shows the features learned by the first and second layer of our deep model. Here, we review the details of architectures used in the experiments.

DeepSets We use a network comprising of 3 permutation-equivariant layers with 256 channels followed by max-pooling over the set structure. The resulting vector representation of the set is then fed to a fully connected layer with 256 units followed by a 40-way softmax unit. We use Tanh activation at all layers and dropout on the layers after set-max-pooling (i.e. two dropout operations) with 50% dropout rate. Applying dropout to permutation-equivariant layers for point-cloud data deteriorated the performance. We observed that using different types of permutation-equivariant layers (see Appendix C) and as few as 64 channels for set layers changes the result by less than 5% in classification accuracy.

For the setting with 5000 particles, we increase the number of units to 512 in all layers and randomly rotate the input around the z-axis. We also randomly scale the point-cloud by s ∼ U(.8, 1./.8). For this setting only, we use Adamax [54] instead of Adam and reduce learning rate from .001 to .0005.

Graph convolution. For each point-cloud instance with 1000 particles, we build a sparse K-nearest neighbor graph and use the three point coordinates as input features. We normalized all graphs at the preprocessing step. For direct comparison with set layer, we use the exact architecture of 3 graph-convolution layer followed by set-pooling (global graph pooling) and dense layer with 256 units. We use exponential linear activation function instead of Tanh as it performs better for graphs. Due to over-fitting, we use a heavy dropout of 50% after graph-convolution and dense layers. Similar to dropout for sets, all the randomly selected features are simultaneously dropped across the graph nodes. the We use a mini-batch size of 64 and Adam for optimization where the learning rate is .001 (the same as that of permutation-equivariant counter-part).

Despite our efficient sparse implementation using Tensorflow, graph-convolution is significantly slower than the set layer. This prevented a thorough search for hyper-parameters and it is quite possible that better hyper-parameter tuning would improve the results that we report here.

<table><tr><td>model</td><td>instance size</td><td>representation</td><td>accuracy</td></tr><tr><td>DeepSets + transformation (ours)</td><td>5000 × 3</td><td>point-cloud</td><td>90 ± .3%</td></tr><tr><td>DeepSets (ours)</td><td>1000 × 3</td><td>point-cloud</td><td>87 ± 1%</td></tr><tr><td>Deep-Sets w. pooling only (ours)</td><td>1000 × 3</td><td>point-cloud</td><td>83 ± 1%</td></tr><tr><td>DeepSets (ours)</td><td>100 × 3</td><td>point-cloud</td><td>82 ± 2%</td></tr><tr><td>KNN graph-convolution (ours)</td><td>1000 × (3 + 8)</td><td>directed 8-regular graph</td><td>58 ± 2%</td></tr><tr><td>3DShapeNets [25]</td><td>303</td><td>voxels (using convolutional deep belief net)</td><td>77%</td></tr><tr><td>DeepPano [20]</td><td>64 × 160</td><td>panoramic image (2D CNN + angle-pooling)</td><td>77.64%</td></tr><tr><td>VoxNet [26]</td><td>323</td><td>voxels (voxels from point-cloud + 3D CNN)</td><td>83.10%</td></tr><tr><td>MVCNN [21]</td><td>164 × 164 × 12</td><td>multi-vew images (2D CNN + view-pooling)</td><td>90.1%</td></tr><tr><td>VRN Ensemble [27]</td><td>323</td><td>voxels (3D CNN, variational autoencoder)</td><td>95.54%</td></tr><tr><td>3D GAN [28]</td><td>643</td><td>voxels (3D CNN, generative adversarial training)</td><td>83.3%</td></tr></table>

Table 6: Classification accuracy and the (size of) representation used by different methods on the ModelNet40 dataset.

Tab. 6 compares our method against the competition.<sup>12</sup> Note that we achieve our best accuracy using 5000 × 3 dimensional representation of each object, which is much smaller than most other methods. All other techniques use either voxelization or multiple view of the 3D object for classification. Interestingly, variations of view/angle-pooling, as in [20, 21], can be interpreted as set-pooling where the class-label is invariant to permutation of different views. The results also shows that using fully-connected layers with set-pooling alone (without max-normalization over the set) works relatively well.

We see that reducing the number of particles to only 100, still produces comparatively good results. Using graph-convolution is computationally more challenging and produces inferior results in this setting. The results using 5000 particles is also invariant to small changes in scale and rotation around the z-axis.

![](images/de14ea59d2f5fa0b9cd7f822dd62794b792da98e5294aec07fd4bde997ad7860.jpg)  
Figure 13: Each box is the particle-cloud maximizing the activation of a unit at the firs (top) and second (bottom) permutation-equivariant layers of our model. Two images of the same column are two different views of the same point-cloud.

Features. To visualize the features learned by the set layers, we used Adamax [54] to locate 1000 particle coordinates maximizing the activation of each unit.<sup>13</sup> Activating the tanh units beyond the second layer proved to be difficult. 13 shows the particle-cloud-features learned at the first and second layers of our deep network. We observed that the first layer learns simple localized (often cubic) point-clouds at different $( x , y , z )$ locations, while the second layer learns more complex surfaces with different scales and orientations.

## I Set Anomaly Detection

Our model has 9 convolution layers with $3 \times 3$ receptive fields. The model has convolution layers with 32, 32, 64 feature-maps followed by max-pooling followed by 2D convolution layers with 64, 64, 128 feature-maps followed by another max-pooling layer. The final set of convolution layers have 128, 128, 256 feature-maps, followed by a max-pooling layer with pool-size of 5 that reduces the output dimension to batch-size. $M \times 2 5 6 .$ , where the set-size $M = \mathrm { { 1 6 } }$ . This is then forwarded to three permutation-equivariant layers with 256, 128 and 1 output channels. The output of final layer is fed to the Softmax, to identify the outlier. We use exponential linear units [55], drop out with 20% dropout rate at convolutional layers and 50% dropout rate at the first two set layers. When applied to set layers, the selected feature (channel) is simultaneously dropped in all the set members of that particular set. We use Adam [54] for optimization and use batch-normalization only in the convolutional layers. We use mini-batches of 8 sets, for a total of 128 images per batch.

![](images/e4fb5ae22f5445661fff80579a7693645fef35231fac908efefd6450d4981511.jpg)  
Figure 14: Each row shows a set, constructed from CelebA dataset, such that all set members except for an outlier, share at least two attributes (on the right). The outlier is identified with a red frame. The model is trained by observing examples of sets and their anomalous members, without access to the attributes. The probability assigned to each member by the outlier detection network is visualized using a red bar at the bottom of each image.

![](images/3744acd048c33916bfd7ea87d20f453a4e3142bc345f054a1cb206a286f78ea5.jpg)  
Figure 15: Each row of the images shows a set, constructed from CelebA dataset images, such that all set members except for an outlier, share at least two attributes. The outlier is identified with a red frame. The model is trained by observing examples of sets and their anomalous members and without access to the attributes. The probability assigned to each member by the outlier detection network is visualized using a red bar at the bottom of each image. The probabilities in each row sum to one.