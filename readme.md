## Personality	

- 一组精神特质（类型）
    - 解释和预测思想，感觉和方式的模式 行为
    - 随着时间和环境保持相对稳定

- 人与人之间的差异来源
    - 影响关系，工作，学习等的成功概率
    - 解释35％的生活满意度差异

- 比较：收入（4％），就业（4％），婚姻状况（1％到4％）

- 检测与欺骗行为相互作用，意见，情感等

## Application

- 人格检测

- 精准广告

- 自适应接口和机器人行为
    - 音乐，风格，色调，色彩等

- 心理诊断，法医学
    - 精神病患者，大规模杀人犯，欺凌者，受害者
    - 抑郁，自杀倾向
    
- 人力资源管理

- 文学科学研究，社会心理学，社会语言学等

- Learning the inherent attributes of dialogues explicitly is another way to improve the diversity of dialogues and ensures the consistency. Among different attributes, topic and personality are widely explored. 

## Dataset

- Question-Answer Datasets

    - The WikiQA corpus: A publicly available set of question and sentence pairs, collected and annotated for research on open-domain question answering. In order to reflect the true information need of general users, we used Bing query logs as the question source. Each question is linked to a Wikipedia page that potentially has the answer.

    - Yahoo Language Data: This page features manually curated QA datasets from Yahoo Answers from Yahoo.

    - TREC QA Collection: TREC has had a question answering track since 1999. In each track, the task was defined such that the systems were to retrieve small snippets of text that contained an answer for open-domain, closed-class questions.

- Customer Support Datasets

    - Ubuntu Dialogue Corpus: Consists of almost one million two-person conversations extracted from the Ubuntu chat logs, used to receive technical support for various Ubuntu-related problems. The full dataset contains 930,000 dialogues and over 100,000,000 words

    - Relational Strategies in Customer Service Dataset: A collection of travel-related customer service data from four sources. The conversation logs of three commercial customer service IVAs and the Airline forums on TripAdvisor.com during August 2016.

    - Customer Support on Twitter: This dataset on Kaggle includes over 3 million tweets and replies from the biggest brands on Twitter.

- Dialogue Datasets

    - Semantic Web Interest Group IRC Chat Logs: This automatically generated IRC chat log  is available in RDF, back to 2004, on a daily basis, including time stamps and nicknames.

    - Cornell movie-dialogs corpus: This corpus contains a large metadata-rich collection of fictional conversations extracted from raw movie scripts.

        - 220,579 conversational exchanges between 10,292 pairs of movie characters.
        - Involves 9,035 characters from 617 movies.
        - In total 304,713 utterances.
        - Link: http://www.cs.cornell.edu/~cristian/Cornell_MovieDialogs_Corpus.html.

    - ConvAI2 Dataset: The dataset contains more than 2000 dialogues for PersonaChat task where human evaluators recruited via the crowd sourcing platform Yandex. Toloka chatted with bots submitted by teams.

    - Santa Barbara Corpus of Spoken American English: This dataset includes approximately 249,000 words of transcription, audio, and timestamps at the level of individual intonation units.

    - The NPS Chat Corpus: This corpus consists of 10,567 posts out of approximately 500,000 posts gathered from various online chat services in accordance with their terms of service.

    - Maluuba goal-oriented dialogue: Open dialogue dataset where the conversation aims at accomplishing a task or taking a decision – specifically, finding flights and a hotel. The dataset contains complex conversations and decision-making covering 250+ hotels, flights, and destinations.

- Multilingual Chatbot Datasets

    - NUS Corpus: This corpus was created for social media text normalization and translation. It is built by randomly selecting 2,000 messages from the NUS English SMS corpus and then translated into formal Chinese.

    - EXCITEMENT data sets: These datasets, available in English and Italian, contain negative feedbacks from customers where they state reasons for dissatisfaction with a given company.

## Word Embedding

- Glove: Global vectors for word representation

## Personality Generation in Text

1. An Embodied Dialogue System with Personality and Emotions
> 一种具有个性和情绪的嵌入式对话系统 </br>
> **价值不大** </br>
> Date: 2010 

2. Varying Personality in Spoken Dialogue with a Virtual Human
> 与虚拟人口语对话中的人格变化 </br>
> **价值不大** </br>
> Date: 2009 

3. PERSONAGE: Personality Generation for Dialogue
> PERSONAGE: 对话个性生成器 </br>
> 使用决策树识别句子当中的性格:内向还是外向 </br>
> Date: 2007

4. Using Linguistic Cues for the Automatic Recognition of Personality in Conversation and Text
> 采用语言线索实现对会话及文本中个性的自动识别 </br>
> 使用句子中的词语和决策树技术对作者的性格进行识别:内向还是外向 </br>
> Date: 2007

5. Personality Modeling in Dialogue Systems 
> 对话系统中个性建模 </br>
> 使用句子中的词语和决策树技术对作者的性格进行识别:内向还是外向 </br>
> Date: 2008

6. A Persona-Based Neural Conversation Model
> 基于角色的神经会话模型 </br>
> Seq2Seq模型 使用LSTM对单个角色的对话进行学习 </br>
> Date: 2016

7. Assigning Personality/Identity to a Chatting Machine for Coherent Conversation Generation
> 为聊天机器分配个性/身份以进行连续对话生成 </br>
> gave the system an identity with profile so that the system can answer personalized question consistently. </br>
> 为机器人分配固有属性，从问答数据、微博数据等中学习根据被分配的固有属性进行回答。 </br>
> 缺点：需要处理和标记的数据太多，并且只针对属性相关的问答进行优化。 </br>
> Date: 2017

8. A Neural Chatbot with Personality
> 具有个性的神经聊天机器人 </br>
> 通过对不同角色的对话进行单独建模，学习该角色的个性信息，从而产生具有个性的机器人。 </br>
> 有源代码 </br>
> Date: 2017

9. Personalizing Dialogue Agents: I have a dog, do you have pets too?
> 个性化对话代理：我有一只狗，你也养宠物吗？</br>
> 将给定的个人简介信息作为条件，利用正在交谈的人的信息来预测下一句对话。</br>
> 已知对方的角色对结果影响不大。因为人们趋向于谈自己感兴趣的事情。</br>
- 采用信息检索模型，监督嵌入模型构建基线模型
- 添加个人简介信息的排序记忆网络模型
- Key-Value个人简介记忆网络
- Seq2Seq
- 生成个人简介记忆网络
> Date: 2018

10. A Persona-Based Neural Conversation Model
> 基于人物角色的神经网络对话模型 </br>
> Based on LSTM and individual's tweets </br>
> Date: 2016

11. Personality for Your Chatbot with Recurrent Neural Networks
> https://towardsdatascience.com/personality-for-your-chatbot-with-recurrent-neural-networks-2038f7f34636 </br>
> Date: 2017

12. Topic Aware Neural Response Generation
> 在生成对话中加入主题信息 </br>
> They noticed that people often associate their dialogues with topically related concepts and create their responses according to these concepts. They used Twitter LDA model to get the topic of the input, fed topic information and input representation into a joint attention module and generated a topic-related response.  </br>
> Date: 2016

13. Domain Aware Neural Dialog System
> 在生成对话中加入领域信息 </br>
> It  made a more thorough generalization of the problem. They classified each utterance in the dialogue into one domain, and generated the domain and content of next utterance accordingly. </br>
> Date: 2017

14. Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation
> It jointly modeled the high-level coarse tokens sequence and the dialogue generation explicitly, where the coarse tokens sequence aims to exploit high-level semantics. They exploited nouns and activity-entity for the coarse sequence representation. </br>
> Date: 2016

15. Emotional chatting machine: Emotional conversation generation with internal and external memory
> Added emotion embedding into a generative model and achieved good performance in perplexity. </br>
> Date: 2017

16. Affective Neural Response Generation
> Enhanced the model of producing emotionally rich responses from three aspects: incorporating cognitive engineered affective word embeddings, augmenting the loss objective with an affectconstrained objective function, and injecting affective dissimilarity in diverse beam-search inference procedure. </br>
> Date: 2017

17. Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models
> Date: 2016

18. Neural Personalized Response Generation as Domain Adaptation
> further took the information of addressee into consideration to create a more realistic chatbot. Since the training data comes from different speakers with inconsistency. </br>
> Date: 2017

19. Personalizing a Dialogue System with Transfer Reinforcement Learning
> used transfer reinforcement learning to eliminate inconsistencies. </br>
> Date: 2016

## Personality Detection in Text

1. 25 Tweets to Know You: A New Model to Predict Personality with Social Media
> 25条微博就能了解你: 一种在社交媒体中预测用户个性的全新模型 </br>
> 使用词嵌入和高斯回归的方法对推文作者的性格进行预测 </br>
> Date: 2017

2. Exploring personality prediction from text on social media: A literature review
> 于社交媒体文本中探索用户的个性的预测方法: 一个文献综述 </br>
> 简单的论文数量，使用数据类型，调查用户数量进行了统计 </br>
> Date: 2017

3. PROFILING A SET OF PERSONALITY TRAITS OF TEXT AUTHOR: WHAT OUR WORDS REVEAL ABOUT US
> 剖析文本作者的个性特征: 词汇的使用是揭露我们个性的 </br>
> 根据具有不同词性的词语的使用频率、词汇多样性等信息，对用户的自毁行为倾向进行预测 </br>
> Date: 2016

4. Personality Detection by Analysis of Twitter Profiles
> 通过分析Twitter个人信息进行个性检测 </br>
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

## Dialogue Systems

1. A Survey on Dialogue Systems: Recent Advances and New Frontiers
> 对话系统调查：近期进展与前沿 </br>
> Date: 2017

2. Deep Reinforcement Learning for Dialogue Generation
> 使用深度强化学习生成对话 </br>
> Date: 2016

3. Adversarial Learning for Neural Dialogue Generation
> 使用对抗网络生成对话 </br>
> Date: 2017

4. Dialogue Learning With Human-In-The-Loop
> 基于人类循环的对话学习 </br>
> Date: 2016

## Chatbot Evaluation

1. Automatic Evaluation of Neural Personality-based Chatbots
> 基于人格的神经网络聊天机器人的自动评估 </br>
> Date: 2018

2. Why people use chatbots?
> 为什么人们要使用聊天机器人? </br>
> 该文章能够作为聊天机器人的设计指南 </br>
> 部分原因： Categories which involved: Productivity, Entertainment, Social/relational, Novelty/Curiosity. </br>
> Date: 2017

## Proposals

- Visualization and Understanding Personality in Dialogue

- Chatting under User's Personality: A Intelligent Neural Chatbot 

- A Survey on Personalizing Dialogue Agents