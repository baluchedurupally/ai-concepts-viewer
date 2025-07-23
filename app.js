import React, { useState } from 'react';

// Main App component
const App = () => {
  // Initial data for all AI concepts
  const initialConcepts = [
    {
      title: "Azure AI Document Intelligence",
      definition: "An AI service that uses machine learning to identify and extract key-value pairs, text, and tables from documents. It helps automate data entry, processing, and analysis from various document types, including forms, receipts, invoices, and custom documents.",
      usage: "Programmatically extract structured data from unstructured or semi-structured documents.",
      useCase: "Automating invoice processing, digitizing healthcare records, extracting data from legal contracts, and processing tax forms.",
      example: "A company uses Azure AI Document Intelligence to automatically extract customer names, order numbers, and line items from incoming purchase orders, reducing manual data entry and speeding up order fulfillment."
    },
    {
      title: "Azure AI Face",
      definition: "A service that provides AI algorithms to detect, recognize, and analyze human faces in images and videos. It can detect facial features, identify individuals, and analyze attributes like age, emotion, and head pose.",
      usage: "Face detection, face verification, face identification, and facial attribute analysis.",
      useCase: "User authentication, content moderation, security monitoring, and personalized user experiences.",
      example: "A mobile banking app uses Azure AI Face for facial verification during login, allowing users to securely access their accounts without typing a password."
    },
    {
      title: "Azure AI Language",
      definition: "A cloud-based service that provides advanced natural language processing (NLP) capabilities. It allows developers to build intelligent applications that understand and analyze text, including sentiment analysis, key phrase extraction, entity recognition, and language detection.",
      usage: "Extract insights from text, understand natural language, and build language-aware applications.",
      useCase: "Customer feedback analysis, content categorization, chatbot development, and legal document review.",
      example: "An e-commerce platform uses Azure AI Language to analyze customer reviews, identifying common positive and negative sentiments about specific products and extracting key phrases related to product features."
    },
    {
      title: "Azure AI Personalizer",
      definition: "An AI service that helps create rich, personalized experiences for users. It uses reinforcement learning to learn user preferences in real-time, recommending the most relevant content or actions to optimize a defined reward metric.",
      usage: "Personalize content, recommendations, advertisements, and user interfaces based on individual user behavior.",
      useCase: "Dynamic content delivery on websites, personalized product recommendations, adaptive learning platforms, and customized news feeds.",
      example: "A news website uses Azure AI Personalizer to dynamically arrange articles on its homepage based on a user's past reading habits and interactions, aiming to maximize engagement and time spent on the site."
    },
    {
      title: "Azure AI Vision",
      definition: "A cloud-based service that provides advanced image processing and analysis capabilities. It allows developers to build intelligent applications that can understand the content of images, including object detection, image classification, facial recognition, and optical character recognition (OCR).",
      usage: "Analyze images and videos, extract information, and understand visual content.",
      useCase: "Content moderation, image tagging, visual search, and accessibility features for the visually impaired.",
      example: "A social media platform uses Azure AI Vision to automatically detect and tag objects in user-uploaded photos, making them more searchable and accessible. It also uses it for content moderation to identify inappropriate images."
    },
    {
      title: "Azure Automated ML",
      definition: "Automates the iterative process of machine learning model development. It helps data scientists, analysts, and developers build ML models with high accuracy and efficiency. AutoML automates tasks like feature engineering, algorithm selection, hyperparameter tuning, and model validation.",
      usage: "Quickly and efficiently train high-quality machine learning models without extensive manual intervention.",
      useCase: "Rapid prototyping of ML solutions, developing baseline models, and for users with limited ML expertise to build predictive models.",
      example: "A business analyst wants to predict customer churn but lacks deep ML expertise. They use Azure Automated ML to automatically experiment with different algorithms and hyperparameters on their customer data, quickly generating a highly accurate churn prediction model."
    },
    {
      title: "Azure Cognitive Search",
      definition: "A search-as-a-service cloud solution that provides full-text search and AI capabilities over heterogeneous content. It allows developers to build rich search experiences by indexing data from various sources and optionally enriching it with AI skills like natural language processing and image analysis.",
      usage: "Create powerful search solutions for websites, applications, and enterprise data.",
      useCase: "Product catalogs, document search, internal knowledge bases, and e-commerce site search.",
      example: "A large corporation uses Azure Cognitive Search to create an internal knowledge base that allows employees to quickly find relevant information across thousands of documents, presentations, and emails, even if the search query uses different terminology than the content."
    },
    {
      title: "Azure Form Recognizer",
      definition: "The previous name for Azure AI Document Intelligence. (See Azure AI Document Intelligence for details).",
      usage: "(See Azure AI Document Intelligence)",
      useCase: "(See Azure AI Document Intelligence)",
      example: "(See Azure AI Document Intelligence)"
    },
    {
      title: "Azure OpenAI Service",
      definition: "Provides access to OpenAI's powerful language models, including GPT-3, Codex, and Embeddings, with the security, compliance, and enterprise-grade capabilities of Azure. It allows developers to integrate advanced generative AI capabilities into their applications.",
      usage: "Natural language understanding, content generation, code generation, summarization, and more.",
      useCase: "Building intelligent chatbots, content creation tools, code assistants, and sophisticated search engines.",
      example: "A marketing agency uses Azure OpenAI Service to generate various ad copy variations for a new product campaign, leveraging the model's ability to create compelling and diverse text based on a few input prompts."
    },
    {
      title: "Best Suited Compute Resources for various ML's",
      definition: "Refers to selecting the optimal computational infrastructure (e.g., CPU, GPU, specialized AI accelerators) for different machine learning tasks, considering factors like data size, model complexity, training time requirements, and inference latency.",
      usage: "Optimizing performance and cost for training and deploying machine learning models.",
      useCase: "Training deep learning models (GPUs), running traditional ML algorithms (CPUs), deploying low-latency inference models (edge devices), or processing large datasets (distributed computing).",
      example: "For training a large image recognition deep learning model, a data scientist would choose Azure VMs with powerful GPUs (e.g., NC-series or ND-series) due to their parallel processing capabilities, which are essential for neural network training. For deploying a simple regression model for real-time predictions, a less powerful CPU-based Azure App Service might suffice."
    },
    {
      title: "Classification",
      definition: "A supervised machine learning task where the model learns to assign input data points to predefined categories or classes. It's used for predicting discrete labels.",
      usage: "Categorizing data into distinct groups.",
      useCase: "Email spam detection (spam/not spam), image recognition (cat/dog/bird), medical diagnosis (disease/no disease), and sentiment analysis (positive/negative/neutral).",
      example: "A machine learning model is trained on a dataset of customer transactions to classify new transactions as either 'fraudulent' or 'legitimate' based on various features like transaction amount, location, and frequency."
    },
    {
      title: "Clustering",
      definition: "An unsupervised machine learning task that groups a set of objects in such a way that objects in the same group (called a cluster) are more similar to each other than to those in other groups. It discovers inherent structures in data without predefined labels.",
      usage: "Discovering natural groupings or patterns in data.",
      useCase: "Customer segmentation, anomaly detection, document clustering, and image segmentation.",
      example: "A retail company uses clustering to segment its customer base into distinct groups based on their purchasing behavior (e.g., high-value frequent shoppers, occasional bargain hunters), allowing for targeted marketing campaigns."
    },
    {
      title: "Context Generation",
      definition: "In AI, especially in conversational AI or natural language processing, involves creating or inferring relevant contextual information to improve understanding or generate more coherent and relevant responses. It can refer to synthesizing a broader understanding from given inputs.",
      usage: "Enhancing the coherence and relevance of AI-generated text or responses by building a richer understanding of the ongoing interaction or topic.",
      useCase: "Chatbots maintaining conversational flow, improving summarization by considering surrounding text, or providing more relevant search results by understanding the user's intent within a broader context.",
      example: "In a customer support chatbot, if a user asks 'What about the refund?', the context generation component understands that 'the refund' refers to a previously discussed order, allowing the bot to provide details specific to that order rather than asking for clarification."
    },
    {
      title: "Context Modeling",
      definition: "The process of representing and managing contextual information in AI systems, particularly for understanding user intent, managing dialogue state, and providing personalized experiences. It involves capturing and updating relevant details from ongoing interactions or environmental factors.",
      usage: "Maintaining state and relevance in AI interactions, especially in conversational agents.",
      useCase: "Chatbots remembering previous turns in a conversation, personalized recommendations based on user history, and adaptive user interfaces.",
      example: "A smart home assistant uses context modeling to understand that when a user says 'Turn off the lights,' they are referring to the lights in the specific room they are currently in, based on sensor data and previous interactions."
    },
    {
      title: "Contextual Mapping",
      definition: "Involves relating different pieces of information or entities within a given context to understand their relationships and overall meaning. This is often crucial in NLP for disambiguation and deeper understanding.",
      usage: "Disambiguating meaning, understanding relationships between entities, and improving information retrieval by considering the surrounding context.",
      useCase: "Semantic search, advanced entity linking, and improving machine translation accuracy by understanding how words relate to each other in a sentence.",
      example: "In the sentence 'The bank decided to raise its interest rates,' contextual mapping helps determine that 'bank' refers to a financial institution, not a river bank, based on its relationship with 'interest rates.'"
    },
    {
      title: "Conversational Retrieval",
      definition: "Refers to the process of finding and presenting relevant information within a dialogue system based on natural language queries, often considering the context of the ongoing conversation. It goes beyond simple keyword matching to understand user intent and conversational history.",
      usage: "Providing accurate and contextually relevant information in interactive conversational agents.",
      useCase: "Customer service chatbots answering complex queries, intelligent personal assistants, and internal knowledge base bots.",
      example: "A chatbot assisting with IT support understands a user's query about 'printer issues' in the context of their previous statements about a specific model of printer, retrieving troubleshooting steps relevant only to that printer."
    },
    {
      title: "Data Parsing",
      definition: "The process of analyzing a string of symbols or data elements, often in a structured format, to extract meaningful components and transform them into a more usable or structured representation. In AI, it's often a precursor to further analysis.",
      usage: "Converting raw data into a structured format for machine learning models or analysis.",
      useCase: "Extracting specific fields from log files, converting JSON or XML data into relational tables, and processing text for NLP tasks.",
      example: "A script parses a CSV file of customer data, separating each line into individual fields like 'Name,' 'Email,' and 'Purchase History' based on comma delimiters, making the data ready for a machine learning model."
    },
    {
      title: "Deep Learning",
      definition: "A subfield of machine learning that uses artificial neural networks with multiple layers (deep neural networks) to learn complex patterns and representations from large amounts of data. It has significantly advanced capabilities in areas like image recognition, natural language processing, and speech recognition.",
      usage: "Solving complex AI problems that involve large, unstructured datasets.",
      useCase: "Image classification, natural language understanding, speech recognition, autonomous driving, and medical image analysis.",
      example: "Training a deep convolutional neural network (CNN) to identify different breeds of dogs from images, where the network automatically learns hierarchical features from low-level edges to high-level dog characteristics."
    },
    {
      title: "Dialog Orchestration",
      definition: "The process of managing the flow and progression of a conversation within a dialogue system. It involves determining the next appropriate action or response based on user input, system state, and predefined conversational logic or AI models.",
      usage: "Guiding and controlling the interaction flow in conversational AI systems.",
      useCase: "Complex customer service bots, virtual assistants handling multi-turn requests, and interactive voice response (IVR) systems.",
      example: "In a travel booking bot, dialog orchestration ensures that after a user specifies their destination, the bot then prompts for travel dates, then number of passengers, and so on, following a logical sequence to complete the booking."
    },
    {
      title: "Dialog Structuring",
      definition: "Involves designing the framework and logical organization of a conversation. This includes defining conversation turns, user intents, system responses, and the overall narrative flow to create coherent and effective human-AI interactions.",
      usage: "Designing the blueprint for conversational AI applications.",
      useCase: "Defining the user journey for a chatbot, structuring interactive tutorials, and designing the flow of an automated phone system.",
      example: "Before building a banking chatbot, developers perform dialog structuring by mapping out potential user intents (e.g., 'check balance,' 'transfer funds'), the questions the bot needs to ask to fulfill each intent, and the corresponding responses."
    },
    {
      title: "Document Intelligence",
      definition: "A broader term encompassing the use of AI to understand, process, and extract information from documents. This often involves combining techniques like OCR, natural language processing, and machine learning. (See Azure AI Document Intelligence for the specific Azure service.)",
      usage: "Automating document processing, data extraction, and information management.",
      useCase: "Invoice automation, contract analysis, legal discovery, and healthcare record management.",
      example: "A legal firm uses document intelligence solutions to automatically extract key clauses, dates, and parties from hundreds of legal contracts, significantly speeding up contract review."
    },
    {
      title: "Entity Recognition",
      definition: "A natural language processing task that identifies and classifies named entities in text into predefined categories such as person names, organizations, locations, dates, monetary values, etc. (Also known as Named Entity Recognition - NER).",
      usage: "Extracting specific, structured information from unstructured text.",
      useCase: "Information extraction, content categorization, search engines, and data anonymization.",
      example: "In the sentence 'Satya Nadella visited Microsoft headquarters in Redmond yesterday,' entity recognition would identify 'Satya Nadella' as a Person, 'Microsoft' as an Organization, 'Redmond' as a Location, and 'yesterday' as a Date."
    },
    {
      title: "Face Attribute Analysis",
      definition: "(Part of Azure AI Face) Involves detecting and analyzing various characteristics of a detected face, beyond just identification. This includes attributes like age, gender, emotion (e.g., happiness, sadness), head pose, eyeglasses, and facial hair.",
      usage: "Extracting demographic and emotional insights from facial images.",
      useCase: "Personalized advertising, audience analysis, content moderation (e.g., detecting inappropriate expressions), and user experience analytics.",
      example: "A marketing research firm analyzes a video of focus group participants using face attribute analysis to understand their emotional reactions (e.g., surprise, joy) to a new product advertisement at different points in the video."
    },
    {
      title: "Face Detection",
      definition: "(Part of Azure AI Face) The technology that identifies the presence and location of human faces in an image or video. It typically returns bounding box coordinates for each detected face. It does not identify *who* the person is.",
      usage: "Locating faces within visual data as a preliminary step for further facial analysis.",
      useCase: "Auto-focus in cameras, blurring faces for privacy, counting people in a crowd, and as a prerequisite for face recognition or attribute analysis.",
      example: "A security camera system uses face detection to trigger an alert only when a human face is detected in a specific area, rather than being triggered by general motion."
    },
    {
      title: "Face Verification",
      definition: "(Part of Azure AI Face) The process of confirming whether two faces belong to the same person, or whether a face belongs to a claimed identity. It's a 1:1 comparison.",
      usage: "Authenticating a user's identity based on their face.",
      useCase: "Unlocking a smartphone, identity verification for online services, secure access control systems, and two-factor authentication.",
      example: "A user logs into an online banking app by taking a selfie, and the app uses face verification to compare the selfie to a previously stored image of the user, confirming their identity."
    },
    {
      title: "Facial Recognition",
      definition: "A broader term that often encompasses both face detection and face identification (determining *who* a person is from a database of known faces). It refers to the technology's ability to identify or verify a person's identity using their face.",
      usage: "Identifying individuals, confirming identities, and tracking people.",
      useCase: "Law enforcement investigations, airport security, access control, and user authentication.",
      example: "A law enforcement agency uses facial recognition to identify a suspect from security camera footage by comparing their face against a database of known individuals."
    },
    {
      title: "Feature vs Label",
      definition: "In machine learning, a **feature** (also known as an attribute or independent variable) is an individual measurable property or characteristic of a phenomenon being observed. A **label** (also known as a target or dependent variable) is the output or outcome that a machine learning model is trained to predict.",
      usage: "Defining the input and output variables for a supervised machine learning task.",
      useCase: "Any supervised learning problem, such as classification or regression.",
      example: "In a model predicting house prices:\n* **Features:** Number of bedrooms, square footage, neighborhood, year built.\n* **Label:** The selling price of the house."
    },
    {
      title: "Image Classification",
      definition: "A computer vision task where a machine learning model is trained to assign a single class label to an entire input image based on its content.",
      usage: "Categorizing images into predefined groups.",
      useCase: "Organizing photo libraries, content filtering, medical image analysis (e.g., classifying X-rays as cancerous or not), and quality control in manufacturing.",
      example: "A model is trained to classify images of animals into categories like 'cat,' 'dog,' or 'bird.' When given a new image of a dog, it outputs 'dog.'"
    },
    {
      title: "Information Retrieval",
      definition: "The science of searching for information within documents, within documents themselves, or for metadata about documents, as well as searching relational databases and the World Wide Web. In AI, it often involves understanding user queries and finding relevant content.",
      usage: "Building search engines, recommender systems, and question-answering systems.",
      useCase: "Web search, enterprise search, academic literature search, and knowledge base lookups.",
      example: "When a user types a query into Google, the search engine uses information retrieval techniques to find and rank relevant web pages from its vast index."
    },
    {
      title: "Interactive Conversation",
      definition: "Refers to a dialogue between a human and an AI system that simulates natural human conversation, allowing for multi-turn exchanges, context understanding, and dynamic responses.",
      usage: "Building engaging and effective conversational AI applications.",
      useCase: "Customer support chatbots, virtual personal assistants, educational tutors, and interactive storytelling.",
      example: "A user asks a chatbot, 'What's the weather like today?' The bot responds with the current weather. The user then asks, 'What about tomorrow?', and the bot understands the context to provide tomorrow's weather without needing the full query again."
    },
    {
      title: "Key Phrase Extraction",
      definition: "A natural language processing task that identifies the most important noun phrases or concepts from a block of text. These phrases often summarize the main points of the text.",
      usage: "Summarizing documents, tagging content, and identifying important topics.",
      useCase: "Analyzing customer feedback for recurring themes, categorizing news articles, and generating keywords for search engine optimization.",
      example: "From the sentence 'The new smartphone features an incredible camera, long-lasting battery, and a vibrant OLED display,' key phrase extraction might identify 'incredible camera,' 'long-lasting battery,' and 'vibrant OLED display.'"
    },
    {
      title: "Language Modeling",
      definition: "The task of assigning a probability to a sequence of words or characters. It's fundamental to many NLP tasks as it learns the statistical relationships between words, enabling the prediction of the next word in a sequence.",
      usage: "Predicting the next word in a sequence, generating coherent text, and understanding natural language.",
      useCase: "Autocomplete in keyboards, speech recognition, machine translation, and generative AI for text creation.",
      example: "Given the phrase 'The cat sat on the...', a language model predicts that 'mat' or 'couch' are highly probable next words, while 'car' is less probable."
    },
    {
      title: "Lexical Mapping",
      definition: "Refers to the process of associating words or lexical units with their corresponding meanings, concepts, or semantic representations. It's crucial for natural language understanding and translation, allowing AI systems to understand the 'what' of language.",
      usage: "Understanding word meanings, synonymy, and semantic relationships between words.",
      useCase: "Machine translation, semantic search, question-answering systems, and information extraction.",
      example: "Lexical mapping helps an AI system understand that 'automobile,' 'car,' and 'vehicle' all refer to the same general concept."
    },
    {
      title: "Multimodal Alignment",
      definition: "Refers to the process of finding correspondences or relationships between different modalities of data (e.g., text, image, audio, video). This allows AI systems to understand how information presented in one modality relates to another.",
      usage: "Building AI systems that can process and understand information from multiple senses.",
      useCase: "Describing images with text, generating video from text descriptions, cross-modal retrieval (e.g., finding images based on text queries), and lip-syncing animations to speech.",
      example: "A model performing multimodal alignment might learn to associate specific objects detected in an image (visual modality) with their corresponding names in a caption (textual modality)."
    },
    {
      title: "Named Entity Recognition",
      definition: "A natural language processing task that identifies and classifies named entities in text into predefined categories such as person names, organizations, locations, dates, monetary values, etc. (Identical to Entity Recognition).",
      usage: "(See Entity Recognition)",
      useCase: "(See Entity Recognition)",
      example: "(See Entity Recognition)"
    },
    {
      title: "Neural Machine Translation",
      definition: "A machine translation approach that uses a large artificial neural network to predict the likelihood of a sequence of words, typically an entire sentence, directly modeling the source and target languages. It often produces more fluent and contextual translations than traditional methods.",
      usage: "Translating text from one human language to another.",
      useCase: "Real-time translation services, document translation, communication across language barriers, and global content localization.",
      example: "Google Translate uses NMT to translate a paragraph from English to Spanish, capturing nuances and grammatical structures more effectively than older rule-based or statistical methods."
    },
    {
      title: "Neural Refinement",
      definition: "Generally refers to the process of improving or fine-tuning the output or performance of an AI model, often a neural network, through subsequent processing steps or additional learning. This can involve making generated content more coherent, accurate, or stylistically appropriate.",
      usage: "Enhancing the quality, accuracy, or naturalness of AI-generated content or model predictions.",
      useCase: "Post-processing generated text for grammatical errors, improving the naturalness of synthesized speech, or refining image generation to be more realistic.",
      example: "After a large language model generates an initial draft of an article, a 'neural refinement' step might use a smaller, specialized neural network to check for factual consistency and stylistic improvements."
    },
    {
      title: "Object Detection",
      definition: "A computer vision task that identifies and locates instances of objects within an image or video. It not only tells you *what* objects are present but also *where* they are by drawing bounding boxes around them.",
      usage: "Identifying specific objects and their positions in visual data.",
      useCase: "Autonomous vehicles (detecting cars, pedestrians, traffic signs), surveillance (identifying intruders), retail analytics (tracking product placement), and quality control in manufacturing.",
      example: "An object detection model analyzes a photo and draws bounding boxes around each car, pedestrian, and traffic light it identifies, labeling each object."
    },
    {
      title: "Opinion Classification",
      definition: "A natural language processing task that categorizes text based on the subjective opinion or stance expressed within it. It's closely related to sentiment analysis but can involve more granular categories of opinion or aspect-based sentiment.",
      usage: "Understanding public opinion, customer feedback, and social media sentiment towards specific topics or entities.",
      useCase: "Product review analysis, political discourse analysis, brand reputation monitoring, and market research.",
      example: "An opinion classification system analyzes product reviews to determine if the review expresses a positive, negative, or neutral opinion specifically about the product's 'battery life' versus its 'camera quality.'"
    },
    {
      title: "Phonetic Enhancement",
      definition: "Generally refers to techniques used in speech processing to improve the clarity, naturalness, or intelligibility of speech by manipulating its phonetic characteristics. This can involve adjusting pronunciation, intonation, or articulation.",
      usage: "Improving the quality and naturalness of synthesized speech or clarifying noisy speech signals.",
      useCase: "Text-to-speech systems for more realistic voices, speech recognition in challenging acoustic environments, and voice conversion applications.",
      example: "In a text-to-speech system, phonetic enhancement might ensure that proper nouns or foreign words are pronounced correctly based on their phonetic transcription, leading to a more natural-sounding output."
    },
    {
      title: "Primary functions of Test Dataset in ML",
      definition: "The test dataset is a subset of the original dataset that is held out during the model training process. Its primary functions are to evaluate the generalization ability of the trained model on unseen data and to provide an unbiased estimate of the model's performance.",
      usage: "Assessing the performance, accuracy, and generalization capabilities of a trained machine learning model.",
      useCase: "Final evaluation of a model before deployment, comparing different models to select the best one, and ensuring the model does not overfit to the training data.",
      example: "After training a spam detection model on 80% of an email dataset, the remaining 20% (the test dataset) is used to see how well the model identifies new, unseen spam and non-spam emails."
    },
    {
      title: "Regression",
      definition: "A supervised machine learning task where the model learns to predict a continuous numerical output value based on input features.",
      usage: "Predicting continuous quantities.",
      useCase: "Predicting house prices, forecasting stock prices, estimating a person's age based on their photo, and predicting sales figures.",
      example: "A linear regression model is trained to predict the price of a house (continuous value) based on features like its size, number of bedrooms, and location."
    },
    {
      title: "Relationship Extraction",
      definition: "A natural language processing task that identifies semantic relationships between named entities in text. For example, it can identify that 'Satya Nadella is the CEO of Microsoft,' recognizing a 'CEO of' relationship between two entities.",
      usage: "Building knowledge graphs, populating databases, and understanding complex information from unstructured text.",
      useCase: "Analyzing legal documents for contractual relationships, news article analysis to understand who did what to whom, and populating knowledge bases for question-answering systems.",
      example: "From the sentence 'Marie Curie discovered radium,' relationship extraction identifies a 'discovered' relationship between 'Marie Curie' (Person) and 'radium' (Chemical Element)."
    },
    {
      title: "Sentiment Analysis",
      definition: "(Also known as opinion mining) A natural language processing technique used to determine the emotional tone behind a body of text. It categorizes text as positive, negative, or neutral sentiment.",
      usage: "Understanding the emotional content of text.",
      useCase: "Analyzing customer reviews, social media monitoring, brand reputation management, and understanding public opinion on products or services.",
      example: "A company uses sentiment analysis on tweets mentioning their product to quickly identify widespread negative feedback after a new feature release."
    },
    {
      title: "Spacial Analysis",
      definition: "This appears to be a common misspelling of 'Spatial Analysis.' (See Spatial Analysis for details).",
      usage: "(See Spatial Analysis)",
      useCase: "(See Spatial Analysis)",
      example: "(See Spatial Analysis)"
    },
    {
      title: "Spatial Analysis",
      definition: "The process of examining data that has a geographical or spatial component to identify patterns, relationships, and trends. In the context of AI, it often involves interpreting spatial relationships from images or sensor data.",
      usage: "Understanding spatial relationships and patterns in visual or geographical data.",
      useCase: "Analyzing traffic patterns, optimizing delivery routes, understanding crowd movement in surveillance, and smart city planning.",
      example: "In a retail store, spatial analysis of camera footage might track customer paths to identify high-traffic areas and optimize store layout."
    },
    {
      title: "Speech Recognition",
      definition: "The technology that enables the recognition and translation of spoken language into text. (See Speech-to-Text for the more common and direct term in AI/ML contexts.)",
      usage: "Converting audio speech into written text.",
      useCase: "Voice assistants, dictation software, transcribing meetings, and voice commands for devices.",
      example: "When you speak a command to your smartphone, speech recognition converts your spoken words into text that the device can then process."
    },
    {
      title: "Speech Synthesis",
      definition: "The artificial production of human speech. A computer system used for this purpose is called a speech synthesizer, and can convert text into spoken language. (See Text-to-Speech for the more common and direct term in AI/ML contexts.)",
      usage: "Generating human-like speech from text.",
      useCase: "Voice assistants, audiobooks, accessibility tools for the visually impaired, and automated public announcements.",
      example: "A navigation app uses speech synthesis to provide spoken turn-by-turn directions to a driver."
    },
    {
      title: "Speech Translation",
      definition: "The process of translating spoken words from one language into spoken words of another language in real-time. It combines speech recognition, machine translation, and speech synthesis.",
      usage: "Facilitating real-time communication across language barriers.",
      useCase: "Live international conferences, cross-lingual customer support, and personal communication when traveling.",
      example: "During a video call, a speech translation system listens to someone speaking in Japanese, translates it into English, and then outputs the English translation as spoken audio to the other participant."
    },
    {
      title: "Speech-to-Text",
      definition: "An AI technology that accurately transcribes spoken language into written text. It is a fundamental component of many voice-enabled applications.",
      usage: "Converting spoken words into digital text.",
      useCase: "Voice assistants (e.g., Siri, Alexa), dictation software, transcribing meetings and interviews, and adding captions to videos.",
      example: "A journalist uses a speech-to-text service to transcribe an hour-long interview, saving significant time compared to manual transcription."
    },
    {
      title: "Syntactic Analysis",
      definition: "(Also known as parsing) A natural language processing task that analyzes the grammatical structure of sentences. It determines how words are grouped together and how they relate to each other in a sentence to form phrases and clauses.",
      usage: "Understanding the grammatical correctness and structure of natural language.",
      useCase: "Grammar checking, machine translation, information extraction (by understanding subject-verb-object relationships), and improving language understanding for chatbots.",
      example: "Syntactic analysis of the sentence 'The quick brown fox jumps over the lazy dog' would identify 'The quick brown fox' as a noun phrase and 'jumps over the lazy dog' as a verb phrase, showing their grammatical relationship."
    },
    {
      title: "Text Adaption",
      definition: "Refers to modifying or transforming text to suit a specific purpose, audience, or format. This can involve simplification, summarization, style transfer, or localization, often using natural language generation techniques.",
      usage: "Customizing text content for different contexts or user needs.",
      useCase: "Summarizing long documents for quick reading, simplifying complex medical texts for patients, translating content for different locales, or adapting a news article for a younger audience.",
      example: "An AI system takes a complex legal document and performs text adaptation to generate a simplified version that is easier for a non-legal professional to understand, without losing critical information."
    },
    {
      title: "Text Analytics",
      definition: "A broad term encompassing various techniques and processes used to derive meaningful insights and patterns from unstructured text data. It includes tasks like sentiment analysis, key phrase extraction, entity recognition, and topic modeling.",
      usage: "Extracting valuable information, insights, and patterns from large volumes of textual data.",
      useCase: "Customer feedback analysis, competitive intelligence, risk management, and scientific literature review.",
      example: "A company uses text analytics to process millions of customer support tickets, identifying common issues, recurring complaints, and emerging product problems."
    },
    {
      title: "Text Generation",
      definition: "The task of automatically creating coherent and contextually relevant natural language text. This can range from generating individual words or sentences to entire articles or stories.",
      usage: "Creating new text content.",
      useCase: "Chatbot responses, content creation (e.g., articles, marketing copy), creative writing (e.g., poems, scripts), and data augmentation.",
      example: "An AI model generates a product description for a new electronic gadget based on a few keywords provided by a marketing team."
    },
    {
      title: "Text Summarization",
      definition: "A natural language processing task that aims to create a concise and coherent summary of a longer text document while retaining the most important information. This can be extractive (picking sentences from the original) or abstractive (generating new sentences).",
      usage: "Condensing information, providing quick overviews, and making long documents more digestible.",
      useCase: "Summarizing news articles, legal documents, research papers, and meeting transcripts.",
      example: "An AI tool takes a 50-page research paper and generates a one-page summary highlighting the main findings, methodology, and conclusions."
    },
    {
      title: "Text-To-Speech",
      definition: "An AI technology that converts written text into human-like spoken audio. It is also known as Speech Synthesis.",
      usage: "Generating spoken audio from textual input.",
      useCase: "Voice assistants, audiobooks, accessibility features for the visually impaired, IVR systems, and creating voiceovers for videos.",
      example: "A GPS navigation system uses text-to-speech to read out street names and directions."
    },
    {
      title: "Voice Adaptation",
      definition: "Refers to the process of customizing or tuning a voice model (often in speech synthesis) to match a specific speaker's voice characteristics (e.g., timbre, intonation, speaking style) or to adapt to a particular acoustic environment.",
      usage: "Personalizing synthesized voices or making speech recognition robust to different voices/accents.",
      useCase: "Creating personalized voice assistants that sound like the user, generating audio in the voice of a specific celebrity, or fine-tuning speech recognition for individual speakers with unique speech patterns.",
      example: "A user records a short audio sample of their own voice, and a Text-to-Speech system then adapts its synthetic voice to mimic the user's unique vocal qualities for subsequent generated speech."
    }
  ];

  // State to hold the concepts data, allowing it to be mutable
  const [concepts, setConcepts] = useState(initialConcepts);
  // State to keep track of the currently selected concept
  const [selectedConcept, setSelectedConcept] = useState(null);
  // State for the search query
  const [searchQuery, setSearchQuery] = useState('');
  // State to manage editing mode
  const [isEditing, setIsEditing] = useState(false);
  // State to temporarily hold changes during editing
  const [editedConcept, setEditedConcept] = useState(null);

  // Filtered concepts based on search query
  const filteredConcepts = concepts.filter(concept =>
    concept.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Function to handle selecting a concept
  const handleSelectConcept = (concept) => {
    setSelectedConcept(concept);
    setIsEditing(false); // Exit editing mode when a new concept is selected
    setEditedConcept(null); // Clear any pending edits
  };

  // Function to initiate editing
  const handleEdit = () => {
    setIsEditing(true);
    setEditedConcept({ ...selectedConcept }); // Create a copy for editing
  };

  // Function to handle changes in editable fields
  const handleChange = (e) => {
    const { name, value } = e.target;
    setEditedConcept(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Function to save changes
  const handleSave = () => {
    setConcepts(prevConcepts =>
      prevConcepts.map(concept =>
        concept.title === editedConcept.title ? editedConcept : concept
      )
    );
    setSelectedConcept(editedConcept); // Update the displayed selected concept
    setIsEditing(false);
    setEditedConcept(null);
  };

  // Function to cancel editing
  const handleCancel = () => {
    setIsEditing(false);
    setEditedConcept(null); // Discard changes
  };

  return (
    <div className="font-sans antialiased bg-gray-100 min-h-screen p-4 md:p-8 flex flex-col items-center">
      {/* Tailwind CSS CDN script */}
      <script src="https://cdn.tailwindcss.com"></script>
      {/* Inter font CDN link */}
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet" />

      {/* Main container for the app */}
      <div className="bg-white rounded-xl shadow-lg w-full max-w-7xl flex flex-col md:flex-row overflow-hidden">
        {/* Left Panel: List of Concepts with Search and Scrolling */}
        <div className="w-full md:w-[30%] bg-gray-50 p-4 md:p-6 border-b md:border-b-0 md:border-r border-gray-200 flex flex-col">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 pb-2 border-b border-gray-300 rounded-md">AI Concepts</h2>

          {/* Search Input */}
          <input
            type="text"
            placeholder="Search concepts..."
            className="w-full p-2 mb-4 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />

          {/* Scrollable list of concepts - Fixed height for 10 items */}
          <ul className="space-y-2 overflow-y-auto pr-2" style={{ maxHeight: '520px' }}> {/* Applied fixed height here */}
            {filteredConcepts.length > 0 ? (
              filteredConcepts.map((concept, index) => (
                <li key={index}>
                  <button
                    onClick={() => handleSelectConcept(concept)}
                    className={`w-full text-left p-3 rounded-lg transition-all duration-200 ease-in-out
                      ${selectedConcept && selectedConcept.title === concept.title
                        ? 'bg-blue-600 text-white shadow-md'
                        : 'bg-white text-gray-700 hover:bg-blue-100 hover:text-blue-700'
                      } focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50`}
                  >
                    {concept.title}
                  </button>
                </li>
              ))
            ) : (
              <li className="text-gray-500 text-center py-4">No concepts found.</li>
            )}
          </ul>
        </div>

        {/* Right Panel: Concept Details with Edit/Save/Cancel */}
        <div className="w-full md:w-[70%] p-4 md:p-6">
          {selectedConcept ? (
            <div className="bg-white p-6 rounded-xl shadow-md border border-gray-200">
              <div className="flex justify-between items-center mb-4 pb-2 border-b-2 border-blue-500 rounded-md">
                <h1 className="text-3xl font-extrabold text-gray-900">
                  {selectedConcept.title}
                </h1>
                {/* Edit/Save/Cancel Buttons */}
                {!isEditing ? (
                  <button
                    onClick={handleEdit}
                    className="flex items-center px-4 py-2 bg-blue-500 text-white rounded-lg shadow-md hover:bg-blue-600 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
                  >
                    {/* Edit Icon (inline SVG) */}
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.38-2.828-2.828z" />
                    </svg>
                    Edit
                  </button>
                ) : (
                  <div className="flex space-x-2">
                    <button
                      onClick={handleSave}
                      className="flex items-center px-4 py-2 bg-green-500 text-white rounded-lg shadow-md hover:bg-green-600 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50"
                    >
                      {/* Save Icon (inline SVG) */}
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Save
                    </button>
                    <button
                      onClick={handleCancel}
                      className="flex items-center px-4 py-2 bg-red-500 text-white rounded-lg shadow-md hover:bg-red-600 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50"
                    >
                      {/* Cancel Icon (inline SVG) */}
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M9.707 10l-3.354 3.354a1 1 0 001.414 1.414L11 11.414l3.354 3.354a1 1 0 001.414-1.414L12.414 10l3.354-3.354a1 1 0 00-1.414-1.414L11 8.586 7.646 5.232a1 1 0 00-1.414 1.414L9.707 10z" clipRule="evenodd" />
                      </svg>
                      Cancel
                    </button>
                  </div>
                )}
              </div>

              {/* Definition Section */}
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-700 mb-2">Definition:</h3>
                {isEditing ? (
                  <textarea
                    name="definition"
                    value={editedConcept.definition}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[100px]"
                  />
                ) : (
                  <p className="text-gray-600 leading-relaxed">{selectedConcept.definition}</p>
                )}
              </div>

              {/* Usage Section */}
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-700 mb-2">Usage:</h3>
                {isEditing ? (
                  <textarea
                    name="usage"
                    value={editedConcept.usage}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[80px]"
                  />
                ) : (
                  <p className="text-gray-600 leading-relaxed">{selectedConcept.usage}</p>
                )}
              </div>

              {/* Use Case Section */}
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-gray-700 mb-2">Use Case:</h3>
                {isEditing ? (
                  <textarea
                    name="useCase"
                    value={editedConcept.useCase}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[120px]"
                  />
                ) : (
                  <p className="text-gray-600 leading-relaxed">{selectedConcept.useCase}</p>
                )}
              </div>

              {/* Example Section */}
              <div>
                <h3 className="text-xl font-semibold text-gray-700 mb-2">Example:</h3>
                {isEditing ? (
                  <textarea
                    name="example"
                    value={editedConcept.example}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[150px]"
                  />
                ) : (
                  <p className="text-gray-600 leading-relaxed whitespace-pre-wrap">{selectedConcept.example}</p>
                )}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500 text-lg">
              Select a concept from the left to view its details.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
