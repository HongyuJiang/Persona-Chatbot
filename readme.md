# Personalizing Chatbot - A Note

## Personality	
- 一组精神特质（类型）
    - 解释和预测思想，感觉和方式的模式 行为
    - 随着时间和环境保持相对稳定
- 人与人之间的差异来源
    - 影响关系，工作，学习等的成功概率
    - 解释35％的生活满意度差异
- 比较：收入(4％)，就业(4％)，婚姻状况(1％到4％)
- 检测与欺骗行为相互作用，意见，情感等

### Mesurement Index

**PAD emotional state model**: pleasure (P), arousal (A), and dominance (D) 

**Big Five personality traits**: openness to experience(O), conscientiousness(C), extraversion(E), agreeableness(A), and neuroticism(N), 

## Application

- 人格检测
    - Using Linguistic Inquiry and Word Count dictionary
- 精准广告
- 自适应接口和机器人行为
    - 音乐，风格，色调，色彩等
- 心理诊断，法医学
    - 精神病患者，大规模杀人犯，欺凌者，受害者
    - 抑郁，自杀倾向
- 人力资源管理
- 文学科学研究，社会心理学，社会语言学等
- 明确的学习对话中的固有属性是提高对话多样性和连贯性的一种方法，在不同的属性中，主题和个性被广泛的探索。 

## Dialogue Systems

**Defination**: A dialogue system, or conversational agent (CA), is a computer system intended to converse with a human with a coherent structure. Dialogue systems have employed text, speech, graphics, haptics, gestures, and other modes for communication on both the input and output channel. (Definetion from Wiki)

### Why people use chatbots?
> 为什么人们要使用聊天机器人? </br>
> 该文章能够作为聊天机器人的设计指南 </br>
> Some reasons：Productivity, Entertainment, Social/relational, Novelty/Curiosity. </br>
> Date: 2017

### A Survey on Dialogue Systems: Recent Advances and New Frontiers
> 对话系统调查：近期进展与前沿 </br>
> Date: 2017


### Goal-oriented

#### Pipeline

- Natural Language Understanding
- Dialogue State Tracer
- Dialogue Tictac Learning
- Natural Language Generation 

#### End to End

1. End-to-End Knowledge-Routed Relational Dialogue System for
Automatic Diagnosis
> 基于知识导向的对话诊断系统 </br>
> Date: 2019
 

-----


### Open domain

#### Generation

**Pros:**
- End-to-end learning
- Safe responses
- Easy to be context-aware, emotional and controllable

**Cons:**
- Hard to evaluate
- Boring and disfluent responses
- Require experienced developers
- UNK


1. Deep Reinforcement Learning for Dialogue Generation
> 使用深度强化学习生成对话 </br>
> Date: 2016

2. Adversarial Learning for Neural Dialogue Generation
> 使用对抗网络生成对话 </br>
> Date: 2017

3. Context-Aware Dialog RE-RANKING FOR TASK-ORIENTED DIALOG SYSTEMS
> 基于上下文信息匹配程度和候选回复的得分对候选回复进行排序 </br>
> Open-Domain End2End </br>
> Date: 2018

#### Retrieval-based

**Pros:**
- Diverse and fluent responses
- Fluent responses
- Flexible system
- Easy to evaluation(L2R)

**Cons:**
- Random response
- Bundled with query-response pairs
- Difficult to be context-aware

- Convolutional Neural Network Architectures for Matching Natural Language Sentences
> 通过在视觉和语音中调整卷积策略，提出用于匹配两个句子的卷积神经网络模型 </br>
> Date: 2014

- Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots
> They propose a sequential matching network (SMN) to keep relationships among utterances and important contextual information in concatenates utterances
> Date: 2018

- DocChat: An Information Retrieval Approach for Chatbot Engines
Using Unstructured Documents
> A novel information retrieval approach for chatbot engines that can leverage unstructured documents, instead of Q-R pairs, to respond to utterances.
> Date: 2016

#### Hybird Method
- AliMe Chat: A Sequence to Sequence and Rerank based Chatbot Engine
> An open-domain chatbot engine that integrates the joint results of Information Retrieval (IR) and Sequence to Sequence (Seq2Seq) based generation models
----

### Diversification

- Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models </br>
> Beam Search: 第一步选择候选词的TopN，此后每次将候选词与此表进行组合，选择候选组合的TopN作为结果 </br>
> Diverse Beam Search: 分多组做Beam Search, 并且保持组间的多样性 </br>
> Date: 2016

- Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models
> 使用了 latent variable 来解决无意义回复(e.g. I don't known)这个问题。

### Supervised Improvement

- Dialogue Learning With Human-In-The-Loop
> 基于人类循环的对话学习 </br>
> 基于强化学习，机器人作出回应后，回答者作出反馈，机器人通过反馈来提高其问答能力。 </br>
> Date: 2016

## Corpus

- Persona-Chat
> Link: https://github.com/facebookresearch/ParlAI/tree/master/parlai/mturk/tasks/personachat/personachat_chat

- Personalized bAbI Dialog
> Link: https://www.dropbox.com/s/4i9u4y24pt3paba/personalized-dialog-dataset.tar.gz?dl=1

- The Persuasion and Personality Corpus
> Link: https://nlds.soe.ucsc.edu/persuasion_persona 

- Question-Answer Datasets

    - The WikiQA corpus
    - Yahoo Language Data
    - TREC QA Collection

- Customer Support Datasets

    - Ubuntu Dialogue Corpus

    - Relational Strategies in Customer Service Dataset

    - Customer Support on Twitter

- Dialogue Datasets

    - Semantic Web Interest Group IRC Chat Logs:

    - Cornell movie-dialogs corpus: This corpus contains a large metadata-rich collection of fictional conversations extracted from raw movie scripts.

        - 220,579 conversational exchanges between 10,292 pairs of movie characters.
        - Involves 9,035 characters from 617 movies.
        - In total 304,713 utterances.
        - Link: http://www.cs.cornell.edu/~cristian/Cornell_MovieDialogs_Corpus.html.

    - ConvAI2 Dataset

    - Santa Barbara Corpus of Spoken American English

    - The NPS Chat Corpus

    - Maluuba goal-oriented dialogue

- Multilingual Chatbot Datasets

    - NUS Corpus: This corpus was created for social media text normalization and translation. It is built by randomly selecting 2,000 messages from the NUS English SMS corpus and then translated into formal Chinese.

    - EXCITEMENT data sets: These datasets, available in English and Italian, contain negative feedbacks from customers where they state reasons for dissatisfaction with a given company.

## Word Embedding

 - word2vec (Mikolov et al., 2013)
 
 - GloVe (Pennington et al., 2014}
 
 - CoVe (Mccann et al., 2017 )

## Topic/Personality in Text

#### How to Make a Digital Personality of Yourself Using Chatbots, Facebook Messages, and Empathy

> Link: https://chatbotsmagazine.com/how-to-make-a-digital-personality-of-yourself-using-chatbots-facebook-and-empathy-8b0c53afa9bd

#### Computerized text analysis
> Link: http://liwc.wpengine.com/

### Traditional Method

1. INDIGO: Interaction with Personality and Dialogue Enabled Robots
> 预设不同的性格给机器人，使其完成不同的任务，例如具有开放性格的机器人将更多地关注用户的请求，而具有高度责任感的机器人将倾向于推荐欣赏展览的最佳路线</br>
> Date: 2008

2. Making Them Remember—Emotional Virtual Characters with Memory
> 从对话中识别用户的情绪，使虚拟人物做出不同的反应</br>
> Date: 2009

3. PERSONAGE: Personality Generation for Dialogue
> PERSONAGE: 对话个性生成器 </br>
> 使用决策树识别句子当中的性格:内向还是外向 </br>
> Date: 2007

4. An Embodied Dialogue System with Personality and Emotions
> 一种具有个性和情绪的嵌入式对话系统 </br>
> 没有给出具体的实现方案，只是给出宏观框架, A Tactical Questioning system </br>
> Date: 2010 

5. Varying Personality in Spoken Dialogue with a Virtual Human
> 与虚拟人口语对话中的人格变化 </br>
> A Tactical Questioning system </br>
> Date: 2009 

4. Using Linguistic Cues for the Automatic Recognition of Personality in Conversation and Text
> 采用语言线索实现对会话及文本中个性的自动识别 </br>
> 使用句子中的词语和决策树技术对作者的性格进行识别:内向还是外向 </br>
> Date: 2007

5. Personality Modeling in Dialogue Systems 
> 对话系统中个性建模 </br>
> 使用句子中的词语和决策树技术对作者的性格进行识别:内向还是外向 </br>
> Date: 2008

---

### Deep Learning Based

#### Personality for Your Chatbot with Recurrent Neural Networks
> https://towardsdatascience.com/personality-for-your-chatbot-with-recurrent-neural-networks-2038f7f34636 </br>

1. A Persona-Based Neural Conversation Model
> 基于角色的神经会话模型 </br>
> Seq2Seq模型 使用LSTM对单个角色的对话进行学习 </br>
> Date: 2016

2. Assigning Personality/Identity to a Chatting Machine for Coherent Conversation Generation
> 为聊天机器分配个性/身份以进行连续对话生成 </br>
> Gave the system an identity with profile so that the system can answer personalized question consistently. </br>
> 为机器人分配固有属性，从问答数据、微博数据等中学习根据被分配的固有属性进行回答。 </br>
> 缺点：需要处理和标记的数据太多，并且只针对属性相关的问答进行优化。 </br>
> Date: 2017

3. A Neural Chatbot with Personality
> 具有个性的神经聊天机器人 </br>
> 通过对不同角色的对话进行单独建模，学习该角色的个性信息，从而产生具有个性的机器人。 </br>
> 有源代码 </br>
> Date: 2017

4. Personalizing Dialogue Agents: I have a dog, do you have pets too?
> 个性化对话代理：我有一只狗，你也养宠物吗？</br>
> 将给定的个人简介信息作为条件，利用正在交谈的人的信息来预测下一句对话。</br>
> 已知对方的角色对结果影响不大。因为人们趋向于谈自己感兴趣的事情。</br>
> - 采用信息检索模型，监督嵌入模型构建基线模型
> - 添加个人简介信息的排序记忆网络模型
> - Key-Value个人简介记忆网络
> - Seq2Seq
> - 生成个人简介记忆网络 

> Date: 2018

5. A Persona-Based Neural Conversation Model
> 基于人物角色的神经网络对话模型 </br>
> Based on LSTM and individual's tweets </br>
> Date: 2016

6. Topic Aware Neural Response Generation
> 在生成对话中加入主题信息 </br>
> 人类的聊天过程中常常围绕在一个显性或者隐性的主题下的概念中，并且回答的内容也根据概念而生成。 </br>
> 该论文使用推特数据结合LDA算法建立对话主题识别模型，将句子的主题信息和输入相结合生成主题相关的回答。 </br>
> Date: 2016

7. Domain Aware Neural Dialog System
> 在生成对话中加入领域信息 </br>
> 将对话中的每一次发言中的主题信息进行识别，并生成回复的主题信息和详细内容</br>
> Date: 2017

8. Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation
> 该论文对高级浅层标记序列和对话生成进行联合建模，其中标记序列旨在挖掘高级语义信息。为了进行粗浅序列表示，他们从对话中发现名词和活动实体。</br>
> Date: 2016

9. Emotional chatting machine: Emotional conversation generation with internal and external memory
> 将情感信息嵌入到对话生成模型中，在困惑度中表现良好. </br>
> Date: 2017

10. Affective Neural Response Generation
> 从三个方面增强了产生具有情绪反应的对话模型：1.结合认知工程的情感词嵌入，2.使用受影响的目标函数增强损失对象，3.在不同的波束搜索推理过程中注入情感差异。 </br>
> Date: 2017

11. Neural Personalized Response Generation as Domain Adaptation
> 进一步考虑了对话接收者的信息，以创建一个更现实的聊天机器人。 由于训练数据来自不同的发言者，能够稳定的产生不同的聊天风格. </br>
> Date: 2017

12. Personalizing a Dialogue System with Transfer Reinforcement Learning
> 使用转移强化学习来增强对话连贯性 </br>
> Date: 2016

13. Social Media Sources for Personality Profiling
> 文本的长度和数量，语法，缩写，拼写和语法错误的不同以及主题的差异会影响预处理的类型和难度，提取正确的文本，模型训练的准确性，训练文本的时间片抽样，以及随时间推移的准确性下降速度 </br>
> Date: 2017

14. 'Ruspersonality': a Russian Corpus for Authorship Profiling and Deception Detection
> 一个剖析作者和欺骗检测俄罗斯语语料库 </br>
> Date: 2016

15. A Context Based Dialog System with a Personality
> End 2 End </br>
> 在Speaker Model 中嵌入人物性格 </br>
> Date: 2018 

16. soc2seq: Social Embedding meets Conversation Model
> 在生成回复的文本模型中考虑了用户的内容偏好和用户交谈的特性 </br>
> Date: 2017

17. User Modeling for Task Oriented Dialogues
> 在生成对话模型中加入用户的目标信息</br> 
> End 2 End,  Hierarchical Seq2Seq </br>
> Date: 2018

## Personality Detection

1. 25 Tweets to Know You: A New Model to Predict Personality with Social Media
> 25条微博就能了解你: 一种在社交媒体中预测用户个性的全新模型 </br>
> 使用词嵌入和高斯回归的方法对推文作者的性格进行预测 </br>
> Date: 2017

2. Exploring personality prediction from text on social media: A literature review
> 于社交媒体文本中探索用户的个性的预测方法: 一个文献综述 </br>
> 简单的论文热点统计，使用数据类型，调查用户数量进行了统计 </br>
> Date: 2017

3. PROFILING A SET OF PERSONALITY TRAITS OF TEXT AUTHOR: WHAT OUR WORDS REVEAL ABOUT US
> 剖析文本作者的个性特征: 词汇的使用是揭露我们个性的 </br>
> 根据具有不同词性的词语的使用频率、词汇多样性等信息，对用户的自毁行为倾向进行预测 </br>
> Date: 2016

4. Personality Detection by Analysis of Twitter Profiles
> 通过分析Twitter个人信息和推文进行个性检测 </br>
> Date: 2017

## Chatbot / NLP

1. Visualizing and Understanding Neural Models in NLP
> 可视化和理解自然语言处理中的神经网络模型 </br>
> Date: 2015

2. A New Chatbot for Customer Service on Social Media
> 一种基于LSTM的社交媒体客户服务聊天机器人 </br>
> Date: 2017

3. Intelligent Chatbot using Deep Learning
> A Master Thesis, A Chatbot based on LSTM </br>
> Date: 2017

## Chatbot Evaluation

- Automatic Evaluation of Neural Personality-based Chatbots
> 基于人格的神经网络聊天机器人的自动评估 </br>
> Date: 2018

- Automatic Evaluation of Neural Personality-based Chatbots
> 使用特殊词语的个性得分，对句子和文本的性格得分进行计算，将其加入语言生成模型，对对话生成模型进行在线评估 </br>
> Date: 2018

### Evaluation Method

- Turing Test

- Perplexity
>困惑度（perplexity）的基本思想是：给测试集的句子赋予较高概率值的语言模型较好,当语言模型训练完之后，测试集中的句子都是正常的句子，那么训练好的模型就是在测试集上的概率越高越好，公式如下：

![123](https://latex.codecogs.com/png.latex?P(W)=P(w_{1}w_{2}...w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}w_{2}...w_{N})}})

由公式可知，句子概率越大，语言模型越好，迷惑度越小。

- BLEU
> BLEU 是一种对模型输出和参考答案的 n-gram 进行比较并计算匹配片段个数的方法. 这些匹配片段与它们在上下文 (Context) 中存在的位置无关, 这里仅认为匹配片段数越多, 模型输出的质量越好. BLEU 首先会对语料库中所有语料进行 n-gram 的精度 (Precision) 计算 (这里假设对于每一条文本, 都有且只有一条候选回复):

![123](https://latex.codecogs.com/gif.latex?P_{n}(r,\hat{r})=\frac{\sum_{k}&space;min(h(k,r),&space;h(k,r_{i}))}{\sum_{k}h(k,r_{i})})

- METEOR
> METEOR 矩阵会在候选答案与目标回复之间产生一个明确的分界线 (这个分界线是基于一定优先级顺序确定的, 优先级从高到低依次是: 特定的序列匹配、同义词、词根和词缀、释义). 有了分界线之后, METEOR 可以把参考答案与模型输出的精度 (Precision) 与召回率 (Recall) 的调和平均值作为结果进行评价. 具体的作法是: 对于一个模型输出 c 与其对应的参考答案 r 的 (c, r)序列 m, METEOR 矩阵值是其精度 Pm 与召回率 Rm 的调和平均值, Pen 是根据已有的正确答案预先计算出的一个惩罚因子, 公式中的 α, β, θ 都是具有默认值的超参数常量.

![123](https://latex.codecogs.com/gif.latex?\small&space;F_{mean}=&space;\frac{P_{m}R_{m}}{\alpha&space;P{m}&plus;(1-\alpha)R_{m}})

![123](https://latex.codecogs.com/gif.latex?\mathrm{Pen}=&space;\gamma(\mathrm{frag})^\theta)

![123](https://latex.codecogs.com/gif.latex?\mathrm{METEOR}&space;=&space;(1-\mathrm{Pen})F_{mean})


- ROUGE
> ROUGE 是一系列用于自动生成文本摘要的评价矩阵, 记为 ROUGE-L, 它是通过对候选句与目标句之间的最长相同子序列 (longest common subsequence, LCS) 计算 F 值 (F-measure) 得到的. LCS 是在两句话中都按相同次序出现的一组词序列, 与 n-gram 不同的是, LCS 不需要保持连续(即在 LCS 中间可以出现其他的词). 公式中 sij 表示与候选回复 ci 对应的第 j 个模型输出, l(ci, sij )表示两者间 LCS 的长度, β 是超参数常量.

![123](https://latex.codecogs.com/gif.latex?R&space;=&space;\underset{j}{max}\frac{l(c_{i},s_{ij})}{\left&space;|&space;s_{ij}&space;\right&space;|})


![123](https://latex.codecogs.com/gif.latex?P&space;=&space;\underset{j}{max}\frac{l(c_{i},s_{ij})}{\left&space;|&space;c_{ij}&space;\right&space;|})


![123](https://latex.codecogs.com/gif.latex?\mathrm{ROUGE}_{\mathrm{L}}(c_{i},&space;s_{i})&space;=&space;\frac{(1&plus;\beta^{2})RP}{R&plus;\beta^{2}P})


## Current weakness 
- 无法灵活调整语言风格
- 缺少根据用户信息动态调整对话策略的能力
- 难以处理用户请求中的歧义项

## Proposals

- Visualization and Understanding Personality in Dialogue

- Chatting under User's Personality: A Intelligent Neural Chatbot 

- A Survey on Personalizing Dialogue Agents

## P.S

>如果Encoder是RNN的话，后输入的单词会稀释之前单词的权重，因此所有的单词并非等权的，Google提出Sequence to Sequence模型时发现把输入句子逆序输入做翻译效果会更好。