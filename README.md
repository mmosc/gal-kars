# GAL-KARS: Exploiting LLMs for Graph Augmentation in Knowledge-Aware Recommender Systems

### Repository for the paper 'GAL-KARS: Exploiting LLMs for Graph Augmentation in Knowledge-Aware Recommender System'


![screenshot](images/fig_a.png)
![screenshot](images/fig_b.png)




### Abstract
In this paper, we propose a recommendation model that exploits a graph augmentation technique based on Large Language Models (LLMs) to enrich the information encoded in its underlying Knowledge Graph (KG). Our work relies on the assumption that the triples encoded in a KG can often be noisy or incomplete, and this may lead to sub-optimal modeling of both the characteristics of items and the users’ preferences. In this setting, graph augmentation can be a suitable solution to improve the quality of the data model and provide users with high-quality recommendations. In this paper, we align with this research line and propose GAL-KARS (Graph Augmentation with LLMs for Knowledge-Aware Recommender Systems). In our framework, we start from a KG, and we design some prompts for querying an LLM and augmenting the graph by incorporating: (a) further features describing the items; (b) further nodes describing the preferences of the users, obtained by reasoning over the items they like. The resulting KG is then passed through a Knowledge Graph Encoder that learns users’ and items’ embeddings based on the augmented KG. These embeddings are finally used to train a recommendation model and provide users with personalized suggestions. As shown in the experimental session, graph augmentation based on LLMs can significantly improve the predictive accuracy of our recommendation model, thus confirming the effectiveness of the model and the validity of our intuitions.