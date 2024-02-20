

classification = {
    'hkunlp/instructor-base': {
        'Banking77Classification': 'Represent the bank purpose for classification: ',
        'EmotionClassification':  'Represent an emotion sentence for classifying if the sentence is positive or not: ',
        'TweetSentimentExtractionClassification': 'Represent a Twitter sentence for classification: ',
        'AmazonCounterfactualClassification': 'Represent an amazon sentence for classifying whether the sentence is counterfactual or not: ',
        'ImdbClassification': 'Represent an amazon review sentence for classifying emotion as positive or negative: ',
        'MassiveIntentClassification':'represent the sentence for classifying its purpose asweather_query, '
                                         'audio_volume_other, iot_cleaning, '
                                         'datetime_query, lists_remove, lists_createoradd, '
                                         'datetime_convert, play_music, iot_hue_lightdim, '
                                         'calendar_remove, iot_coffee, '
                                         'general_greet, alarm_query, calendar_set, '
                                         'recommendation_locations, '
                                         'lists_query or email_query: ',
        'MassiveScenarioClassification':  "Represent a scene for determining the scene as"
                                         "play, alarm, music, iot, audio, takeaway, datetime, recommendation, "
                                         "email, cooking, news or question answering: ",
        'MTOPDomainClassification': 'represent a sentence; ',
        'MTOPIntentClassification': 'Represent the sentence for determining its purpose as question_music, '
                                    'start_shuffle_music, get_call_time, '
                                    'get_reminder_location, is_true_recipes, '
                                    'ignore_call, get_contact_method, '
                                    'update_reminder, delete_alarm, set_default_provider_music, '
                                    'end_call, '
                                    'skip_track_music, create_timer, cancel_message, '
                                    'get_category_event, repeat_all_off_music, get_timer, '
                                    'add_time_timer, resume_music, add_to_playlist_music, update_reminder_location, '
                                    'set_rsvp_interested, pause_timer, update_timer, play_media, replay_music: ',
        'ToxicConversationsClassification': 'Represent a toxicity comment for classifying its toxicity as toxic or non-toxic: ',
        'AmazonPolarityClassification': 'Represent the sentiment comment for retrieving a duplicate sentence: ',
        'AmazonReviewsClassification': 'represent an amazon review sentence: ',
    },
    'hkunlp/instructor-large':{
        'Banking77Classification': 'Represent the bank purpose for classification: ',
        'EmotionClassification':  'Represent an emotion sentence for classifying the emotion: ',
        'TweetSentimentExtractionClassification': 'Represent a Tweet sentence for classification: ',
        'AmazonCounterfactualClassification': 'Represent a counter-factual sentence for classification: ',
        'ImdbClassification': 'Represent a review sentence for classifying emotion as positive or negative: ',
        'MassiveIntentClassification':'Represent the sentence for classifying the purpose as one of qa_maths, takeaway_order, weather_query, '
                                       'audio_volume_other, recommendation_movies, iot_cleaning, qa_stock, '
                                       'iot_hue_lighton, iot_hue_lightchange, alarm_remove, play_radio, '
                                       'transport_taxi, datetime_query, lists_remove, lists_createoradd, '
                                       'datetime_convert, play_music, iot_hue_lightdim, email_querycontact, qa_factoid, '
                                       'cooking_query, music_query, qa_currency, calendar_query, music_settings, '
                                       'music_dislikeness, audio_volume_mute, cooking_recipe, general_joke, play_game, '
                                       'news_query, recommendation_events, music_likeness, audio_volume_down, '
                                       'calendar_remove, iot_coffee, transport_traffic, iot_wemo_off, email_sendemail, '
                                       'iot_hue_lightup, social_query, social_post, iot_hue_lightoff, transport_query, '
                                       'general_greet, play_podcasts, alarm_query, calendar_set, alarm_set, '
                                       'transport_ticket, general_quirky, audio_volume_up, iot_wemo_on, qa_definition, '
                                       'recommendation_locations, play_audiobook, email_addcontact, takeaway_query, '
                                       'lists_query or email_query: ',
        'MassiveScenarioClassification': "Represent the scene for classifying its scene as one of calendar, "
                                         "play, general, alarm, music, iot, audio, takeaway, datetime, recommendation, "
                                         "social, lists, email, transport, cooking, weather, news or qa: ",
        'MTOPDomainClassification': 'Represent a sentence: ',
        'MTOPIntentClassification': 'Represent the sentence for retrieval: ',
        'ToxicConversationsClassification': 'Represent a toxicity comment for classifying its toxicity as toxic or non-toxic: ',
        'AmazonPolarityClassification': 'Represent the sentiment comment for retrieving a duplicate sentence: ',
        'AmazonReviewsClassification 42.12': 'Represent a review sentence for classification: ',
        'AmazonReviewsClassification': 'Represent a review for classification: ',
    },
    'hkunlp/instructor-xl': {
        'Banking77Classification': 'Represent the bank77 purposes for retrieving its bank intent: ',
        'EmotionClassification':  'Represent the amazon emotion sentence for classifying the emotion: ',
        'AmazonCounterfactualClassification': 'Represent Daily casual counter-sentences for categorization as correct-sentences or counter-sentences: ',
        # 'AmazonCounterfactualClassification': 'Represent Daily casual counter-sentences for categorization as correct-sentences or counter-sentences: ',
        'ImdbClassification': 'Represent a review sentence for classifying emotion as positive or negative: ',
        'MassiveIntentClassification':'Represent the sentence for categorizing its task intent as qa_maths, takeaway_order, '
                                       'audio_volume_other, recommendation_movies, iot_cleaning, qa_stock, '
                                      'or recommendation_locations: ',
        'MassiveScenarioClassification': "represent an ms sentence for retrieving its intent: ",
        'MTOPDomainClassification': 'represent a MTO sentence to retrieve the task intention: ',
        'MTOPIntentClassification': 'Represent an mto sentence for retrieving its behind task intention: ',
        'ToxicConversationsClassification': 'Represent a toxicity comment for classifying its toxicity as toxic or non-toxic: ',
        # 'ToxicConversationsClassification': 'Represent toxicity comments for classifying its toxicity as toxic or non-toxic: ',
        'AmazonPolarityClassification': 'Represent the sentiment comment for retrieving a duplicate sentence: ',
        # 'AmazonPolarityClassification': 'Represent the sentiment comment to retrieve a duplicate sentence: ',
        'AmazonReviewsClassification': 'Represent an amazon review sentence to find the emoation; ',
        # 'AmazonReviewsClassification': 'Represent an amazon movie review sentence to categorize the emotion; ',
        'TweetSentimentExtractionClassification': 'Represent Daily-life spoken sentences for categorization; Input: ',
        # 'TweetSentimentExtractionClassification': 'Represent Daily-life spoken expression for classification; Input: ',
    },
}


clustering = {
    'hkunlp/instructor-xl': {
        'TwentyNewsgroupsClustering': 'Represent the news comment for clustering; ',
        'BiorxivClusteringS2S': 'Represent the biological statement for retrieval; ',
        'MedrxivClusteringS2S': 'Represent the Biological statement for clustering biological statements: ',
        'ArxivClusteringP2P': 'Represent the science passage for retrieval: ',
        'ArxivClusteringS2S': 'Represent the Science statements for retrieval: ',
        # 'ArxivClusteringS2S': 'Represent the Scientific statements for retrieval: ',
        'BiorxivClusteringP2P': 'Represent the Biological passage for retrieval: ',
        'MedrxivClusteringP2P': 'Represent the Biological paragraph for retrieval: ',
        # 'MedrxivClusteringP2P': 'Represent the Biological document for retrieval: ',
        'RedditClustering': 'represent the Reddit community title: ',
        # 'RedditClustering': 'represent the Reddit community sentence: ',
        'RedditClusteringP2P': 'represent a Reddit community passage: ',
        'StackExchangeClustering': 'Represent a question for retrieval: ',
        # 'StackExchangeClustering': 'Represent the questions for retrieval: ',
        'StackExchangeClusteringP2P': 'Represent the question and answer passage for retrieving relevant question and answer passages: ',
        # 'StackExchangeClusteringP2P': 'Represent the question and answer passage for retrieving relevant question and answer passages: ',
    },
    # 'hkunlp/instructor-xl original': {
    #     'TwentyNewsgroupsClustering': 'Represent the news comment for clustering; ',
    #     'BiorxivClusteringS2S': 'Represent the biological statement for retrieval; ',
    #     'MedrxivClusteringS2S': 'Represent the Medical statement for retrieving duplicate sentence: ',
    #     'ArxivClusteringP2P': 'Represent the science passage for retrieval: ',
    #     # 'ArxivClusteringS2S 32.05': 'Represent the science statements for retrieval: ',
    #     'ArxivClusteringS2S': 'Represent the Science statements for retrieval: ',
    #     'BiorxivClusteringP2P': 'Represent the Bio-medicine passage for retrieval: ',
    #     'MedrxivClusteringP2P': 'Represent the medicine paragraph for retrieval: ',
    #     'RedditClustering': 'represent the Reddit community title: ',
    #     'RedditClusteringP2P': 'represent a Reddit community passage: ',
    #     'StackExchangeClustering': 'Represent a question for retrieval: ',
    #     'StackExchangeClusteringP2P': 'Represent the question and answer for retrieving duplicate question and answers: ',
    # },
    'hkunlp/instructor-large': {
        'TwentyNewsgroupsClustering': 'Represent the news comment for retrieval: ',
        'BiorxivClusteringS2S': 'Represent the biomedical statement for retrieval: ',
        'MedrxivClusteringS2S': 'Represent the medicine statement for retrieving duplicate sentences: ',
        'ArxivClusteringP2P': 'Represent the science passage for retrieval: ',
        'ArxivClusteringS2S': 'Represent the science statement for retrieval: ',
        'BiorxivClusteringP2P': 'Represent the Bio-medicine passage for retrieval: ',
        'MedrxivClusteringP2P': 'Represent the medicine paragraph for retrieval: ',
        'RedditClustering': 'represent a reddit community title: ',
        'RedditClusteringP2P': 'represent a Reddit community passage: ',
        'StackExchangeClustering': 'Represent the question for retrieval: ',
        'StackExchangeClusteringP2P': 'Represent the question and answer for retrieving duplicate question and answers: ',
    },
    'hkunlp/instructor-base': {
        'TwentyNewsgroupsClustering': 'Represent the news comment for retrieval: ',
        'BiorxivClusteringS2S': 'Represent the biomedical statement for retrieval: ',
        'MedrxivClusteringS2S': 'Represent the medicine statement for retrieving duplicate sentences: ',
        'ArxivClusteringP2P': 'Represent the science passage for retrieval: ',
        'ArxivClusteringS2S': 'Represent the science statement for retrieval: ',
        'BiorxivClusteringP2P': 'Represent the Bio-medicine passage for retrieval: ',
        'MedrxivClusteringP2P': 'Represent the medicine paragraph for retrieval: ',
        'RedditClustering': 'represent a reddit community title: ',
        'RedditClusteringP2P': 'represent a Reddit community passage: ',
        'StackExchangeClustering': 'Represent the question for retrieval: ',
        'StackExchangeClusteringP2P': 'Represent the question and answer for retrieving duplicate question and answers: ',
    },
}


pairclassification = {
    'hkunlp/instructor-base':
        {
            'TwitterSemEval2015': 'Represent the tweet post for retrieving duplicate comments: ',
            'TwitterURLCorpus': 'represent a tweet post for retrieval: ',
            'SprintDuplicateQuestions': 'represent the Sprint post for retrieving duplicate posts: ',
        },
    'hkunlp/instructor-large':
        {
            'TwitterSemEval2015': 'Represent the tweet post for retrieving duplicate comments: ',
            'TwitterURLCorpus': 'represent a tweet post for retrieval: ',
            'SprintDuplicateQuestions': 'represent the Sprint post for retrieving duplicate posts: ',
        },
    'hkunlp/instructor-xl':
        {
            'TwitterSemEval2015': 'Represent the twitter post for retrieving comments: ',
            'TwitterURLCorpus': 'represent a Twitter posts for retrieval: ',
            'SprintDuplicateQuestions': 'represent the Sprint questions for retrieving relevant posts, ',
        },
}

reranking = {
    'hkunlp/instructor-base': {
        'AskUbuntuDupQuestions':
            {
                'query': 'Represent the Ubuntu question for reranking: ',
                'corpus': 'Represent a Ubuntu question for reranking: ',
            },
        'StackOverflowDupQuestions':
            {
                'query': 'Represent the StackOverflow question: ',
                'corpus': 'Represent a StackOverflow question: ',
            },
        'SciDocsRR':
            {
                'query': 'Represent the Scientific title: ',
                'corpus': 'Represent the Scientific document: '
            },
        'MindSmallReranking':
            {
                'query': 'Represent the news query for retrieving articles: ',
                'corpus': 'Represent the news article for retrieval: ',
            },
    },
    'hkunlp/instructor-large': {
        'AskUbuntuDupQuestions':
            {
                'query': 'Represent the Ubuntu question for reranking: ',
                'corpus': 'Represent the Ubuntu question for reranking: ',
            },
        'StackOverflowDupQuestions':
            {
                'query': 'Represent the StackOverflow question: ',
                'corpus': 'Represent the StackOverflow question: ',
            },
        'SciDocsRR':
            {
                'query': 'Represent the Scientific title: ',
                'corpus': 'Represent the Scientific document: '
            },
        'MindSmallReranking':
            {
                'query': 'Represent the news query for retrieving articles: ',
                'corpus': 'Represent the news article for retrieval: ',
            },
    },
    'hkunlp/instructor-xl': {
        'AskUbuntuDupQuestions':
            {
                'query': 'Represent the Ubuntu question to retrieve a duplicate question: ',
                'corpus': 'Represent the Ubuntu question: ',
            },
        'StackOverflowDupQuestions':
            {
                'query': 'Represent the StackOverflow question: ',
                'corpus': 'Represent the StackOverflow question: ',
            },
        'SciDocsRR':
            {
                'query': 'Represent the Science question: ',
                'corpus': 'Represent the Science document: '
            },
        # 'SciDocsRR':
        #     {
        #         'query': 'Represent the Science question to retrieve a supporting document: ',
        #         'corpus': 'Represent the Science document: '
        #     },
        'MindSmallReranking':
            {
                'query': 'Represent the news query for retrieving articles: ',
                'corpus': 'Represent the news article for retrieval: ',
            },
    }
}



sts = {
    'hkunlp/instructor-base': {
        'STS12': 'Represent the statement, ',
        'STS13': 'represent the statement, ',
        'STS14': 'Represent the statement, ',
        'STS15': 'Represent the post, ',
        'STS16': 'Represent the post: ',
        'STS17': 'Represent the sentence for classification: ',
        'STS22': 'represent the statement: ',
        'BIOSSES': 'Represent the Bio-medical statement: ',
        'SICK-R': 'Represent the statement: ',
        'STSBenchmark': 'represent the statement: ',
    },
    'hkunlp/instructor-large': {
        'STS12': 'Represent the statement: ',
        'STS13': 'represent the statement: ',
        'STS14': 'Represent the statement: ',
        'STS15': 'Represent the statement: ',
        'STS16': 'Represent the statement: ',
        'STS17': 'Represent the sentence for classification: ',
        'STS22': 'represent the statement: ',
        'BIOSSES': 'Represent the Bio-medical statement: ',
        'SICK-R': 'Represent the statement: ',
        'STSBenchmark': 'represent the statement: ',
    },
    'hkunlp/instructor-xl': {
        'STS12': 'represent texts, ',
        'STS13': 'represent a casual post, ',
        'STS14': 'Represent a post; ',
        'STS15': 'Represent a posts,,, ',
        # 'STS15': 'Represent a post for classification, ',
        'STS16': 'Represent posts: ',
        'STS17': 'Represent a statement, ',
        'STS22': 'represent the statement: ',
        'BIOSSES': 'represent the Biological statement: ',
        'SICK-R': 'Represent a post: ',
        'STSBenchmark': 'represent posts, ',
    }
}


summarization = {
    'hkunlp/instructor-xl': {
        'SummEval': 'Represent the news statement for retrieval: ',
    },
    'hkunlp/instructor-large': {
        'SummEval': 'Represent the news sentence for retrieval: ',
    },
    'hkunlp/instructor-base': {
        'SummEval': 'Represent the news sentence for retrieval: ',
    }

}




retrieval = {
    'hkunlp/instructor-xl': {
        'ClimateFEVER':
            {
                'query': 'Represent the Climate question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'HotpotQA':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'FEVER':
            {
                'query': 'Represent the fact for retrieving supporting evidence: ',
                'corpus': 'Represent the evidence for retrieval: ',
            },
        'MSMARCO':
            {
                'query': 'Represent the question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'DBPedia':
            {
                'query': 'Represent the Wikipedia questions to retrieve a supporting document: ',
                'corpus': 'Represent the Wikipedia documents for retrieval: ',
            },
        'NQ':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'QuoraRetrieval':
            {
                'query': 'Represent the Quora question to retrieve question: ',
                'corpus': 'Represent the Quora question to retrieve question: ',
            },
        'SCIDOCS':
            {
                'query': 'Represent a Science question for retrieving supporting papers: ',
                'corpus': 'Represent the Science paper: ',
            },
        'TRECCOVID':
            {
                'query': 'Represent the Coronavirus questions to retrieve a supporting document: ',
                'corpus': 'Represent the Coronavirus documents for retrieval: ',
            },
        'Touche2020':
            {
                'query': 'Represent questions: ',
                'corpus': 'Represent arguments: ',
            },
        'SciFact':
            {
                'query': 'Represent the Scientific queries for retrieving a supporting passage: ',
                'corpus': 'represent the scientific paragraph for retrieval: ',
            },
        'NFCorpus':
            {
                'query': 'Represent the nutrition facts to retrieve Public medical articles: ',
                'corpus': 'Represent the Public medical articles for retrieval: ',
            },
        'ArguAna':
            {
                'query': 'Represent Debating conversations to retrieve a counter-argument: ',
                'corpus': 'Represent counter-arguments: ',
            },
        'CQADupstackTexRetrieval':
            {
                'query': 'Represent the question for retrieving answers: ',
                'corpus': 'Represent the answer for retrieval: ',
            },
        'CQADupstackWebmastersRetrieval':
            {
                'query': 'Represent the Webmaster question for retrieving answers: ',
                'corpus': 'Represent the Webmaster answer: ',
            },
        'CQADupstackEnglishRetrieval':
            {
                'query': 'Represent the English question for retrieving documents: ',
                'corpus': 'Represent the English answer for retrieval: ',
            },
        'CQADupstackGamingRetrieval':
            {
                'query': 'Represent the Gaming question for retrieving answers: ',
                'corpus': 'Represent the Gaming answer for retrieval: ',
            },
        'CQADupstackGisRetrieval':
            {
                'query': 'Represent the Gis question for retrieving answers: ',
                'corpus': 'Represent the Gis answer for retrieval: ',
            },
        'CQADupstackUnixRetrieval':
            {
                'query': 'Represent the Unix questions to retrieve a supporting answer: ',
                'corpus': 'Represent the Unix answers for retrieval: ',
            },
        'CQADupstackMathematicaRetrieval':
            {
                'query': 'Represent the Mathematical question for retrieving answers: ',
                'corpus': 'Represent the Mathematical answer for retrieval: ',
            },
        'CQADupstackStatsRetrieval':
            {
                'query': 'Represent the Statistical question for retrieving answers: ',
                'corpus': 'Represent the Statistical answer for retrieval: ',
            },
        'CQADupstackPhysicsRetrieval':
            {
                'query': 'Represent the Physics question for retrieving answers: ',
                'corpus': 'Represent the Physics answer for retrieval: ',
            },
        'CQADupstackProgrammersRetrieval':
            {
                'query': 'Represent the Programming question for retrieving answers: ',
                'corpus': 'Represent the Programming answer for retrieval: ',
            },
        'CQADupstackAndroidRetrieval':
            {
                'query': 'Represent the Android question for retrieving answers: ',
                'corpus': 'Represent the Android answer for retrieval: ',
            },
        'CQADupstackWordpressRetrieval':
            {
                'query': 'Represent the Wordpress question for retrieving answers: ',
                'corpus': 'Represent the Wordpress answer for retrieval: ',
            },
        'FiQA2018':
            {
                'query': 'Represent the finance questions to retrieve a supporting answer: ',
                'corpus': 'Represent the finance answers for retrieval: ',
            },
    },
    'hkunlp/instructor-large':{
        'ClimateFEVER':
            {
                'query': 'Represent the Climate question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'HotpotQA':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'FEVER':
            {
                'query': 'Represent the fact for retrieving supporting evidence: ',
                'corpus': 'Represent the evidence for retrieval: ',
            },
        'MSMARCO':
            {
                'query': 'Represent the question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'DBPedia':
            {
                'query': 'Represent the Wikipedia sentence for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'NQ':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'QuoraRetrieval':
            {
                'query': 'Represent the Quora question for retrieving duplicate questions: ',
                'corpus': 'Represent the Quora question for retrieving duplicate questions: ',
            },
        'SCIDOCS':
            {
                'query': 'Represent a Science question for retrieving supporting papers: ',
                'corpus': 'Represent the Science paper: ',
            },
        'TRECCOVID':
            {
                'query': 'Represent the Coronavirus question for retrieving supporting documents: ',
                'corpus': 'Represent the Coronavirus document for retrieval: ',
            },
        'Touche2020':
            {
                'query': 'Represent a question: ',
                'corpus': 'Represent an argument: ',
            },
        'SciFact':
            {
                'query': 'Represent a Scientific query for retrieving a supporting passage; ',
                'corpus': 'represent the Scientific passage for retrieval; ',
            },
        'NFCorpus':
            {
                'query': 'Represent the Medicine question for retrieving a relevant document: ',
                'corpus': 'Represent the medical document for retrieval: ',
            },
        'ArguAna':
            {
                'query': 'Represent a Debate argument for retrieving a counter-argument: ',
                'corpus': 'Represent a Counter-argument: ',
            },
        'CQADupstackTexRetrieval':
            {
                'query': 'Represent the question for retrieving answers: ',
                'corpus': 'Represent the answer for retrieval: ',
            },
        'CQADupstackWebmastersRetrieval':
            {
                'query': 'Represent the Webmaster question for retrieving answers: ',
                'corpus': 'Represent the Webmaster answer: ',
            },
        'CQADupstackEnglishRetrieval':
            {
                'query': 'Represent the English question for retrieving documents: ',
                'corpus': 'Represent the English answer for retrieval: ',
            },
        'CQADupstackGamingRetrieval':
            {
                'query': 'Represent the Gaming question for retrieving answers: ',
                'corpus': 'Represent the Gaming answer for retrieval: ',
            },
        'CQADupstackGisRetrieval':
            {
                'query': 'Represent the Gis question for retrieving answers: ',
                'corpus': 'Represent the Gis answer for retrieval: ',
            },
        'CQADupstackUnixRetrieval':
            {
                'query': 'Represent the Unix question for retrieving answers: ',
                'corpus': 'Represent the Unix answer for retrieval: ',
            },
        'CQADupstackMathematicaRetrieval':
            {
                'query': 'Represent the Mathematical question for retrieving answers: ',
                'corpus': 'Represent the Mathematical answer for retrieval: ',
            },
        'CQADupstackStatsRetrieval':
            {
                'query': 'Represent the Statistical question for retrieving answers: ',
                'corpus': 'Represent the Statistical answer for retrieval: ',
            },
        'CQADupstackPhysicsRetrieval':
            {
                'query': 'Represent the Physics question for retrieving answers: ',
                'corpus': 'Represent the Physics answer for retrieval: ',
            },
        'CQADupstackProgrammersRetrieval':
            {
                'query': 'Represent the Programming question for retrieving answers: ',
                'corpus': 'Represent the Programming answer for retrieval: ',
            },
        'CQADupstackAndroidRetrieval':
            {
                'query': 'Represent the Android question for retrieving answers: ',
                'corpus': 'Represent the Android answer for retrieval: ',
            },
        'CQADupstackWordpressRetrieval':
            {
                'query': 'Represent the Wordpress question for retrieving answers: ',
                'corpus': 'Represent the Wordpress answer for retrieval: ',
            },
        'FiQA2018':
            {
                'query': 'Represent the finance question for retrieving the supporting answers: ',
                'corpus': 'Represent the finance answer for retrieval: ',
            },
    },
    'hkunlp/instructor-base': {
        'ClimateFEVER':
            {
                'query': 'Represent the Climate question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'HotpotQA':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'FEVER':
            {
                'query': 'Represent the fact for retrieving supporting evidence: ',
                'corpus': 'Represent the evidence for retrieval: ',
            },
        'MSMARCO':
            {
                'query': 'Represent the question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'DBPedia':
            {
                'query': 'Represent the Wikipedia sentence for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'NQ':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'QuoraRetrieval':
            {
                'query': 'Represent the Quora question for retrieving duplicate questions: ',
                'corpus': 'Represent the Quora question for retrieving duplicate questions: ',
            },
        'SCIDOCS':
            {
                'query': 'Represent a Science question for retrieving supporting papers: ',
                'corpus': 'Represent the Science paper: ',
            },
        'TRECCOVID':
            {
                'query': 'Represent the Coronavirus question for retrieving supporting documents: ',
                'corpus': 'Represent the Coronavirus document for retrieval: ',
            },
        'Touche2020':
            {
                'query': 'Represent a question: ',
                'corpus': 'Represent an argument: ',
            },
        'SciFact':
            {
                'query': 'Represent a Scientific query for retrieving a supporting passage; ',
                'corpus': 'represent the Scientific passage for retrieval; ',
            },
        'NFCorpus':
            {
                'query': 'Represent the Medicine question for retrieving a relevant document: ',
                'corpus': 'Represent the medical document for retrieval: ',
            },
        'ArguAna':
            {
                'query': 'Represent the Debate argument for retrieving a counter-argument: ',
                'corpus': 'Represent the Counter debate argument: ',
            },
        'CQADupstackTexRetrieval':
            {
                'query': 'Represent the question for retrieving answers: ',
                'corpus': 'Represent the answer for retrieval: ',
            },
        'CQADupstackWebmastersRetrieval':
            {
                'query': 'Represent the Webmaster question for retrieving answers: ',
                'corpus': 'Represent the Webmaster answer: ',
            },
        'CQADupstackEnglishRetrieval':
            {
                'query': 'Represent the English question for retrieving documents: ',
                'corpus': 'Represent the English answer for retrieval: ',
            },
        'CQADupstackGamingRetrieval':
            {
                'query': 'Represent the Gaming question for retrieving answers: ',
                'corpus': 'Represent the Gaming answer for retrieval: ',
            },
        'CQADupstackGisRetrieval':
            {
                'query': 'Represent the Gis question for retrieving answers: ',
                'corpus': 'Represent the Gis answer for retrieval: ',
            },
        'CQADupstackUnixRetrieval':
            {
                'query': 'Represent the Unix question for retrieving answers: ',
                'corpus': 'Represent the Unix answer for retrieval: ',
            },
        'CQADupstackMathematicaRetrieval':
            {
                'query': 'Represent the Mathematical question for retrieving answers: ',
                'corpus': 'Represent the Mathematical answer for retrieval: ',
            },
        'CQADupstackStatsRetrieval':
            {
                'query': 'Represent the Statistical question for retrieving answers: ',
                'corpus': 'Represent the Statistical answer for retrieval: ',
            },
        'CQADupstackPhysicsRetrieval':
            {
                'query': 'Represent the Physics question for retrieving answers: ',
                'corpus': 'Represent the Physics answer for retrieval: ',
            },
        'CQADupstackProgrammersRetrieval':
            {
                'query': 'Represent the Programming question for retrieving answers: ',
                'corpus': 'Represent the Programming answer for retrieval: ',
            },
        'CQADupstackAndroidRetrieval':
            {
                'query': 'Represent the Android question for retrieving answers: ',
                'corpus': 'Represent the Android answer for retrieval: ',
            },
        'CQADupstackWordpressRetrieval':
            {
                'query': 'Represent the Wordpress question for retrieving answers: ',
                'corpus': 'Represent the Wordpress answer for retrieval: ',
            },
        'FiQA2018':
            {
                'query': 'Represent the finance question for retrieving the supporting answers: ',
                'corpus': 'Represent the finance answer for retrieval: ',
            },
    },
}




task_type_to_instruct = {
    "Classification": classification,
    "Clustering": clustering,
    "PairClassification": pairclassification,
    "Reranking": reranking,
    "Retrieval": retrieval,
    "STS": sts,
    "Summarization": summarization,
}

import pandas as pd

df = pd.DataFrame.from_dict(task_type_to_instruct)


# columns = df.columns()

for scale in ['base', 'large', 'xl']:
    save_path = f'config/instruction_{scale}.json'
    df.loc[f'hkunlp/instructor-{scale}'].to_json(save_path, indent=2)