任务目标：复现当前的图去噪算法，并能够添加进当前框架中使其能够使用脚本跑通


统一使用python3.8，采用pytorch和pyg框架进行搭建，方便后续拓展

项目分为六个模块
1.数据模块（用于数据加载和数据预处理）——data_provider

2.实验类（根据任务划分——节点分类、链路预测、社团检测、图分类等）——其中包含了训练、测试、验证通用方法和模型的创建 ——exp

3.layers （模型需要用到的各层，需要先转换成pytorch等支持的格式）——layers

4.模型类（存放本lib支持的图模型，需要适配框架和修改参数传入方式为命令行）——models
源代码使用的是torch。就使用torch   使用的是其她框架。改为pyg或pytorch

5.工具类 （用于模型的各类工具——如归一化、指标、早停法、采样器等）——utils

6.脚本文件——用于各类模型的启动设置，用户可以直接在此调整模型参数和运行 ——scripts
linux环境使用方法： bash 脚本路径/**.sh  每个模型单独建一个脚本文件夹 

各算法使用方法：(这里需要自己根据模型添加)
dropedge： 数据获取方式：通过采样器获取子图数据，动态数据生成。脚本直接运行

RGIB： 先通过算法提供的脚本静态生成噪声数据，另外提供了标准的训练（不使用RGIB）和RGIB的两种方式


注意：在添加模型前需要学习git，在拉取master分支后在自己的个人分支（可以以组长名称命名）进行开发和模型的建立，
如果需要改动exp、data_provider中的内容，需要将改动的代码与当前master的diff（git平台能看见）发给老师审批后进行上线


模型论文代码对应，此处仅仅列出一部分相关的论文及部分代码，有关图去噪相关的方法都可以自由选择，所选题目需要发给老师审核

1.ND——Network deconvolution as a general method to distinguish direct dependencies in networks
https://compbio.mit.edu/nd/code.html

2.NE——Network enhancement as a general method to denoise weighted biological networks
https://snap.stanford.edu/ne/

3.NetRL——NetRL: Task-Aware Network Denoising via Deep Reinforcement Learning
https://github.com/galina0217/NetRL

4.E-net——Robust Network Enhancement From Flawed Networks
https://github.com/zjunet/E-Net

5.Mask-GVAE——Mask-GVAE: Blind Denoising Graphs via Partition
（肯定有代码）

6.focus——FocusedCleaner: Sanitizing Poisoned Graphs for Robust GNN-Based Node Classification
https://github.com/zhuyulin-tony/focusedcleaner

7.Graph Sanitation——Graph Sanitation with Application to Node Classification


8.NeuralSparse——Robust Graph Representation Learning via Neural Sparsiﬁcation
https://github.com/flyingdoog/PTDNet.

9.PTDNet——Learning to Drop: Robust Graph Neural Network via Topological Denoising

https://github.com/flyingdoog/PTDNet.

10.RGIB——Combating Bilateral Edge Noise for Robust Link Prediction（不可选）
https://github.com/tmlr-group/RGIB.

11.RisKeeper——Value at Adversarial Risk: A Graph Defense Strategy against Cost-Aware Attacks
https://github.com/songwdfu/RisKeeper

12.GRV———Unsupervised Adversarially Robust Representation Learning on Graphs

13.Edmot——EdMot: An Edge Enhancement Approach for Motif-aware Community Detection（不可选）
https://github.com/benedekrozemberczki/EdMot

14.RCD——RobustECD: Enhancement of Network Structure for Robust Community Detection
https://github.com/jjzhou012/RobustECD

15.MetaGC——Robust Graph Clustering via Meta Weighting for Noisy Graphs（不可选）
https://github.com/HyeonsooJo/MetaGC

16.DWSSA——Alleviating over-smoothness for deep Graph Neural Networks（无代码）

17.BGAT-CCRF——BGAT-CCRF: A novel end-to-end model for knowledge graph noise correction

18.RGLC——Rethinking the impact of noisy labels in graph classification: A utility and privacy
perspective
https://github.com/LDer66/RGLC

19.Jaccard——Adversarial examples on graph data: Deep insights into attack and defense
（肯定有代码）

20.RIB——Graph Information Bottleneck
https://github.com/snap-stanford/GIB

21.VIB——Graph Structure Learning with Variational Information Bottleneck
https://github.com/RingBDStack/VIB-GSL

22.PRI——Principle of relevant information for graph sparsiﬁcation
https://github.com/SJYuCNEL/PRI-Graphs

23.DropEdge——DropEdge：TOWARDS DEEP GRAPH CONVOLUTIONAL NETWORKS ON NODE CLASSIFICATION（不可选）
https://github.com/DropEdge/DropEdge.


数据集举例：（每个算法至少选择3～5个数据集实现）
Cora、CiteSeer、Pubmed、Reddit、PPI、Polblogs、FinV、Telecom、polbooks等
更多图相关的数据集可以在https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html中找到