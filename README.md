# 实验报告：基于多种结构洞节点度量指标的Hamsterster社交网络结构洞检测

## 一、实验目的
本实验旨在是使用Python中的networkx库分析给定的社交网络数据集Hamsterster Networks，检测并探讨网络的结构洞，并通过实验报告记录相关分析和发现。

具体实验目的包括：

1. 利用Python中的networkx库，计算多种结构洞节点度量指标，并据此识别网络中的结构洞节点。
2. 增进对不同社交网络结构洞度量指标的理解，以便评估它们在节点结构洞可能性衡量中的重要性。


## 二、实验内容和原理
### 2.1 数据集介绍
本实验使用了名为Hamsterster的社交网络数据集，Hamsterster是一个社交网络，包含1858个节点（成员），它们被划分为36个社区，节点之间共有12534对链接。

网络可视化如下：

![networkx](https://raw.githubusercontent.com/Alanlaye/-Hamsterster/main/Hamsterster/picture/network.png)

### 2.2 实验任务
预测社交网络中的结构洞所在。

### 2.3 结构洞检测原理
结构洞通常表现为社交网络中的一些特定模式或情境，例如社区边界、孤立节点、低密度区域等。在社会关系网络中，两个个体或群体之间如果不存在直接关系，且它们之间不存在间接冗余关系，那么两者之间的间隙就成为结构洞。

本实验基于结构洞理论，综合考量现存的一些能够定量度量结构洞节点的相关指标，计算综合得分进行排序，以确定最具结构洞潜力的节点。

本实验涉及的度量指标包括：

1. **介数中心性（Betweenness Centrality）**：
   - **概念**：介数中心性是一种用于衡量节点在网络中充当中介者或桥梁的程度的指标。节点的介数中心性越高，表示它在网络中的最短路径中起到关键的连接作用。
   - **作用**：高介数中心性的节点通常位于网络的结构洞中，连接不同社区或分割社交网络。

2. **网络约束系数（Constraint）**：
   - **概念**：网络约束系数度量了节点所在社区的内部联系与外部联系之间的平衡。如果节点在社交网络中连接了不同社区，它的网络约束系数可能较低。
   - **作用**：低网络约束系数的节点可能是连接不同社区的结构洞节点。

3. **PageRank**：
   - **概念**：PageRank是一种用于评估节点在网络中重要性的算法，考虑了节点的入度以及连接到节点的其他节点的重要性。
   - **作用**：高PageRank值的节点通常在网络中扮演关键的角色，可能是结构洞节点，特别是在信息传播方面。

4. **局部聚类系数（Local Clustering Coefficient）**：
   - **概念**：局部聚类系数测量了节点邻居之间存在连接的概率，反映了节点所在社交网络中的聚类程度。
   - **作用**：高局部聚类系数的节点通常位于社交网络的密集区域，而低局部聚类系数的节点可能是结构洞节点，连接不同社交群体。

5. **度中心性（Degree Centrality）**：
   - **概念**：度中心性度量了节点的度数，即与节点相连的边的数量。
   - **作用**：高度度中心性的节点在网络中具有更多的连接，通常在信息传播和网络稳定性方面发挥关键作用，是结构洞的重要标志。

6. **社区有效规模（Community Effective Size）**：
   - **概念**：社区有效规模表示节点所在社区的规模，但考虑了社区内部节点之间的连接性质。
   - **作用**：社区有效规模有助于识别社交网络中的核心社区，以及在社交网络中具有高重要性的节点，是结构洞的重要标志。

7. **等级度（Hierarchy）**：
   - **概念**：等级度衡量了节点与其邻居之间的连接层次结构。
   - **作用**：等级度可用于识别节点在社交网络中的连接模式，特别是那些连接不同社区的节点可能是结构洞节点的候选者。


## 三、实验步骤

### 3.1 实验准备
首先,导入实验所需要的包，包括：networkx、matplotlib、community、pandas、IPython（用于最后的表格可视化）

代码如下：

```python
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import pandas as pd
from IPython.display import display, HTML
```

### 3.2 数据加载与图构建
使用networkx库加载Hamsterster数据集，并构建网络图。这包括从数据文件中读取节点和边的信息，并创建一个图表示社交网络。

代码如下：

```python
edge_file = "out.petster-hamster-friend"
G = nx.read_edgelist(edge_file, nodetype=int, comments="%")
```

### 3.3 绘制网络可视化图
为了更好地理解社交网络的结构，实验绘制了一个网络可视化图。这可以帮助观察网络的整体结构和可能的结构洞。（可视化结果见 2.1 数据集介绍）

```python
plt.figure(figsize=(5, 4))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=False, node_size=10)
plt.title("Hamsterster Friends Network")
plt.show()
```

### 3.4 计算度量指标 1 ：介数中心性（Betweenness Centrality）
根据 *《社会网络结构洞节点度量指标比较与分析》（韩忠明，吴杨，谭旭升，刘雯，杨伟杰，2015）* ，介数中心性（Betweenness Centrality） 在众多结指标中对结构洞节点的测量效果最好，所以实验选取介数中心性作为第一个计算的指标。

实验计算出每个节点的介数中心性，又根据介数中心性的直方图呈现的长尾特征，选取0.03作为阈值，将介数中心性大于0.03的节点存入结构洞节点候选列表filtered_nodes。

代码如下：

```python
# 计算所有节点的介数中心性值
centrality_values = list(betweenness_centralities.values())

# 绘制介数中心性分布直方图
plt.figure()
plt.hist(centrality_values, bins=20)
plt.title('Betweenness Centrality Distribution')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Number of Nodes')
plt.show()
```
![BT](https://raw.githubusercontent.com/Alanlaye/-Hamsterster/main/Hamsterster/picture/BT.png)


```python
# 将介数中心性大于0.03的节点存入结构洞节点候选列表filtered_nodes
filtered_nodes = [node for node, centrality in top_20_nodes if centrality >= 0.03]
```

### 3.5 计算度量指标 2 ：网络约束系数（Constraint）

```python
node_constraints = {node: nx.constraint(G, [node]) for node in filtered_nodes}
```

### 3.6 计算度量指标 3 ：PageRank值

```python
pageranks = nx.pagerank(G, personalization={node: 1 for node in filtered_nodes})
```

### 3.7 计算度量指标 4 ：局部聚类系数（Local Clustering Coefficient）

```python
local_clustering_coefficients = nx.clustering(G, nodes=filtered_nodes)
```

### 3.8 计算度量指标 5 ：度中心性（Degree Centrality）

```python
bridge_nodes_degree_centrality = {node: G.degree(node) / (len(G) - 1) for node in filtered_nodes}
```

### 3.9 计算度量指标 6 ：社区有效规模（Community Effective Size）

```python
partition = community_louvain.best_partition(G)

# 计算每个社区的有效规模
community_effective_sizes = {}
for community in set(partition.values()):
    community_nodes = [node for node, comm in partition.items() if comm == community]
    actual_size = sum(G.degree(node) for node in community_nodes) / 2
    potential_size = len(community_nodes) * (len(community_nodes) - 1) / 2
    community_effective_sizes[community] = actual_size / potential_size if potential_size != 0 else 0

# 计算filtered_nodes中每个节点所在社区的有效规模
filtered_effective_sizes = {}
for node in filtered_nodes:
    community = partition[node]  # 找出节点所在的社区
    filtered_effective_sizes[node] = community_effective_sizes[community]

```
### 3.10 计算度量指标 7 ：等级度（Hierarchy）

```python
filtered_hierarchy = {}
for node in filtered_nodes:
    neighbors = set(G.neighbors(node)) 
    if len(neighbors) > 1:  
        connected_neighbors = sum(1 for i in neighbors for j in neighbors if i != j and G.has_edge(i, j))  
        filtered_hierarchy[node] = connected_neighbors / (len(neighbors) * (len(neighbors) - 1)) 
    else:
        filtered_hierarchy[node] = 0 
```

### 3.11 综合得分计算与结果可视化
在实验的最后阶段，实验者对结构洞的候选节点列表进行了综合评估。鉴于实验者在统计和编程方面的知识储备和能力有限，实验采用了相对粗糙的评估方法。

具体来说：实验对与结构洞特征呈负相关的指标结果进行了取反操作，然后将各项指标得分相加。接着，选取得分最高的节点（Node2539），并进一步评估该节点在各项指标上的表现，发现其与结构洞特征的高度契合。最终，得出结论，节点Node2539是该社交网络中的结构洞所在。

```python
# 创建数据框存储指标计算结果
df = pd.DataFrame({
    'Node': filtered_nodes,
    'Betweenness Centrality': [round(betweenness_centralities[node], 4) for node in filtered_nodes],
    'Constraint': [-round(node_constraints[node][node], 4) for node in filtered_nodes],  # 负相关的指标
    'PageRank': [round(pageranks[node], 4) for node in filtered_nodes],
    'Local Clustering Coefficient': [-round(local_clustering_coefficients[node], 4) for node in filtered_nodes],  # 负相关的指标
    'Degree Centrality': [round(bridge_nodes_degree_centrality[node], 4) for node in filtered_nodes],
    'Community Effective Size': [-round(filtered_effective_sizes[node], 4) for node in filtered_nodes],  # 负相关的指标
    'Hierarchy': [round(filtered_hierarchy[node], 4) for node in filtered_nodes]
})

# 设置Node列为索引
df.set_index('Node', inplace=True)

# 计算综合得分
df['Score'] = df.sum(axis=1)

# 生成可视化HTML表格
html = df.to_html(classes='center')
html = html + """
<style>
.center {
    text-align: center;
}
</style>
"""
display(HTML(html))
```

![图片描述](https://raw.githubusercontent.com/Alanlaye/-Hamsterster/main/Hamsterster/picture/result.png)


## 四、实验结果与讨论

### 4.1 实验结果

在本实验中，我们使用多种结构洞节点度量指标对Hamsterster社交网络进行了分析和探索。以下是我们的主要结果：

1. **结构洞节点候选列表**：通过计算介数中心性，我们筛选出了具有高介数中心性的节点，将介数中心性大于0.03的节点存入结构洞节点候选列表filtered_nodes。该列表包含一组可能是结构洞节点的候选者。

2. **多种度量指标的计算**：我们计算了与结构洞节点相关的多种度量指标，包括网络约束系数、PageRank值、局部聚类系数、度中心性、社区有效规模以及等级度。这些指标提供了有关节点在网络中的重要性和与结构洞特征的关联性的信息。

3. **综合得分和最终结论**：为了综合评估结构洞的候选节点，我们使用一种相对粗糙的评估方法。我们对与结构洞特征呈负相关的指标结果进行了取反操作，然后叠加各项指标的得分。最终，我们选取得分最高的节点（Node2539）作为潜在的结构洞节点，并进一步评估该节点在各项指标上的表现。根据高度契合结构洞特征的结果，我们得出结论，节点Node2539是Hamsterster社交网络中的结构洞所在。

### 4.2 讨论

1. **评估方法的限制**：我们采用了一种相对粗糙的评估方法，对与结构洞特征呈负相关的指标进行了取反操作，并叠加各项指标的得分。尽管这种方法在有限的条件下提供了结果，但它并不是最精确的结构洞检测方法。未来的工作可以探索更精确的评估方法。

2. **指标的选择**：我们使用了多种度量指标，但没有详细研究它们在结构洞检测中的权重和相互关系。不同指标可能对不同类型的结构洞具有不同的识别能力，因此更深入的指标分析可能会提高准确性。

总的来说，本实验通过综合评估确定了Hamsterster社交网络中的潜在结构洞节点。然而实验的评估方法有很大改进的空间，希望通过此后的理论学习和编程能力精进，进一步探索更精确的结构洞检测方法。


## 五、参考文献

**参考文献：**

[1] 韩忠明, 吴杨, 谭旭升等. 社会网络结构洞节点度量指标比较与分析[J]. 山东大学学报(工学版), 2015, 45(01): 1-8.

**数据集信息：**

数据集名称：Hamsterster friends network, part of the Koblenz Network Collection

更多关于该网络的信息，请参考以下链接：

[http://konect.cc/networks/petster-hamster-friend](http://konect.cc/networks/petster-hamster-friend)


## 附录：代码和数据
实验完整代码可在以下链接找到：
[实验代码](https://github.com/Alanlaye/-Hamsterster)




