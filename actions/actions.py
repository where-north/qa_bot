# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from collections import OrderedDict
from typing import Any, Dict, List, Text, Optional

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher

import cpca  # cpca是chinese_province_city_area_mapper的简称，可用于处理中文地址
import validators

from actions.utils.coins import CoinDataManager
from actions.utils.request import get
from actions.utils.search import search_anime, AnimalImgSearch
from actions.utils.cqa_es import ElasticSearchBM25 as CQA_ElasticSearchBM25
from actions.utils.dqa_es import ElasticSearchBM25 as DQA_ElasticSearchBM25

import logging
import json
from rasa_sdk.types import DomainDict
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import (
    SlotSet,
    UserUtteranceReverted,
    ConversationPaused,
    EventType,
)
from actions.api import weather
import math
from datetime import datetime, date, timedelta
from .utils.config import *
from transformers import BertForQuestionAnswering, BertTokenizer
from .utils.document_qa_utils import predict

logger = logging.getLogger(__name__)

# 加载QA模型
model = BertForQuestionAnswering.from_pretrained(QA_MODEL_PATH)
model.cuda()
tokenizer = BertTokenizer.from_pretrained(QA_MODEL_PATH, do_lower_case=True)
logger.info('QA模型加载成功！')

CQA_ES = CQA_ElasticSearchBM25(corpus_path=CQA_CORPUS_PATH, index_name=CQA_INDEX_NAME, reindexing=True)

DQA_ES = DQA_ElasticSearchBM25(index_name=DQA_INDEX_NAME, reindexing=True)


# class ActionQueryWeather(Action):
#
#     def name(self) -> Text:
#         return "action_query_weather"
#
#     async def run(self, dispatcher: CollectingDispatcher,
#                   tracker: Tracker,
#                   domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         user_in = tracker.latest_message.get("text")
#         province, city = cpca.transform([user_in]).loc[0, ["省", "市"]]  # 提取text中的省和市
#         city = province if city in ["市辖区", None] else city
#         text = await self.get_weather(await self.get_location_id(city))
#         dispatcher.utter_message(text=text)
#         return []
#
#     @staticmethod
#     async def get_location_id(city):
#         if not QUERY_KEY:
#             raise ValueError("需要获得自己的key。。。看一下官方文档即可。 参考地址: qweather.com")
#         params = {"location": city, "key": QUERY_KEY}
#         res = await get(CITY_LOOKUP_URL, params=params)
#         return res["location"][0]["id"]
#         # return 124
#
#     @staticmethod
#     async def get_weather(location_id):
#         params = {"location": location_id, "key": QUERY_KEY}
#         res = (await get(WEATHER_URL, params=params))["now"]
#         # res = {'code': '200', 'updateTime': '2021-04-12T13:47+08:00', 'fxLink': 'http://hfx.link/2bc1',
#         #        'now': {'obsTime': '2021-04-12T13:25+08:00', 'temp': random.randint(10, 30), 'feelsLike': '19',
#         #                'icon': '305', 'text': '小雨', 'wind360': '315', 'windDir': '西北风', 'windScale': '0',
#         #                'windSpeed': '0', 'humidity': '100', 'precip': '0.1', 'pressure': '1030', 'vis': '3',
#         #                'cloud': '91', 'dew': '16'},
#         #        'refer': {'sources': ['Weather China'], 'license': ['no commercial use']}}
#         # res = res["now"]
#         return f"{res['text']} 风向 {res['windDir']}\n温度: {res['temp']} 摄氏度\n体感温度：{res['feelsLike']}"


class ActionGreetUser(Action):
    """Greets the user with/without privacy policy"""

    def name(self) -> Text:
        return "action_greet_user"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:
        intent = tracker.latest_message["intent"].get("name")
        shown_privacy = tracker.get_slot("shown_privacy")

        if intent == "greet":
            if shown_privacy:
                dispatcher.utter_message(response="utter_greet_noname")
                return []
            else:
                dispatcher.utter_message(response="utter_greet")
                dispatcher.utter_message(response="utter_inform_privacypolicy")
                return [SlotSet("shown_privacy", True)]
        return []


class ActionDefaultAskAffirmation(Action):
    """Asks for an affirmation of the intent if NLU threshold is not met."""

    def name(self) -> Text:
        return "action_default_ask_affirmation"

    def __init__(self) -> None:
        import pandas as pd

        self.intent_mappings = pd.read_csv(INTENT_DESCRIPTION_MAPPING_PATH)
        self.intent_mappings.fillna("", inplace=True)
        self.intent_mappings.entities = self.intent_mappings.entities.map(
            lambda entities: {e.strip() for e in entities.split(",")}
        )

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:

        clear_slots = ['department', 'CQA_has_started', 'DQA_has_started']
        slots_data = domain.get("slots")

        intent_ranking = tracker.latest_message.get("intent_ranking", [])
        if len(intent_ranking) > 1:
            diff_intent_confidence = intent_ranking[0].get(
                "confidence"
            ) - intent_ranking[1].get("confidence")
            if diff_intent_confidence < 0.2:
                intent_ranking = intent_ranking[:2]
            else:
                intent_ranking = intent_ranking[:1]

        first_intent_names = [
            intent.get("name", "")
            if intent.get("name", "") not in ["chitchat", '图书馆服务', '就业指导', '后勤服务', '宿舍服务',
                                              '教学教务', '奖助学金', '补考缓考', '选课事宜', '专业修读',
                                              '学生证件', '校医院', '校园网服务', '研究生招生', '本科生招生',
                                              '迎新服务', '常用联系方式']
            else tracker.latest_message.get("response_selector")
                .get(intent.get("name", ""))
                .get("ranking")[0]
                .get("intent_response_key")
            for intent in intent_ranking
        ]
        if "nlu_fallback" in first_intent_names:
            first_intent_names.remove("nlu_fallback")
        if "greet" in first_intent_names:
            first_intent_names.remove("greet")
        if "/out_of_scope" in first_intent_names:
            first_intent_names.remove("/out_of_scope")
        if "out_of_scope" in first_intent_names:
            first_intent_names.remove("out_of_scope")

        user_query = tracker.latest_message.get("text")

        if len(first_intent_names) > 0:
            message_title = (
                "对不起，我不太理解您的意思🤔，您是想问..."
            )

            entities = tracker.latest_message.get("entities", [])
            entities = {e["entity"]: e["value"] for e in entities}

            entities_json = json.dumps(entities)

            buttons = []
            for intent in first_intent_names:
                button_title = self.get_button_title(intent, entities)
                text = "{'affirmation':{'query': '%s'}}" % button_title
                if len(entities_json) > 2:

                    buttons.append(
                        {"title": text, "payload": f"/{intent}{entities_json}"}
                    )
                else:
                    buttons.append({"title": text, "payload": f"/{intent}"})

            buttons.append({"title": "{'affirmation':{'query': '以上都不是'}}", "payload": "/deny"})

            dispatcher.utter_message(text=message_title, buttons=buttons)
        else:
            message_title = (
                "<div class='msg-text'>对不起，我不太理解您的意思"
                " 🤔 您可以问得再具体一些吗？</div>"
            )
            dispatcher.utter_message(text=message_title)

        return [SlotSet('user_query', user_query)] + [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for
                                                      slot_name in clear_slots]

    def get_button_title(self, intent: Text, entities: Dict[Text, Text]) -> Text:
        default_utterance_query = self.intent_mappings.intent == intent
        utterance_query = (self.intent_mappings.entities == entities.keys()) & (
            default_utterance_query
        )

        utterances = self.intent_mappings[utterance_query].button.tolist()

        if len(utterances) > 0:
            button_title = utterances[0]
        else:
            utterances = self.intent_mappings[default_utterance_query].button.tolist()
            button_title = utterances[0] if len(utterances) > 0 else intent

        return button_title.format(**entities)


class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:
        clear_slots = ['user_query', 'department', 'location', 'CQA_has_started', 'DQA_has_started']
        slots_data = domain.get("slots")
        dispatcher.utter_message(template="utter_stilldontunderstand")
        return [UserUtteranceReverted()] + [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for
                                            slot_name in clear_slots]


def check_last_event(tracker, event_type: Text, skip: int = 2, window: int = 3, slots_data=None) -> bool:
    """
    @param slots_data:
    @param tracker:
    @param event_type: 事件名称
    @param skip: 从最近的倒数第几个事件开始遍历
    @param window: 遍历长度
    @return:
    """
    skipped = 0
    count = 0
    ignore_events = ['action_listen', None]
    ignore_events.extend(list(slots_data.keys()))
    for e in reversed(tracker.events):
        e_name = e.get('name')
        if e_name not in ignore_events:
            count += 1
            if count > window:
                return False
            if e_name == event_type:
                skipped += 1
                if skipped > skip:
                    return True
    return False


class ActionTriggerResponseSelector(Action):
    """Returns the faq utterance dependent on the intent"""

    def name(self) -> Text:
        return "action_trigger_response_selector"

    def __init__(self) -> None:
        import pandas as pd

        self.intent_mappings = pd.read_csv(INTENT_DESCRIPTION_MAPPING_PATH)
        self.intent_mappings.fillna("", inplace=True)
        self.intent_query_mappings = self.intent_mappings.set_index('intent').to_dict()['button']

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> List[EventType]:
        clear_slots = ['department', 'CQA_has_started', 'DQA_has_started']
        slots_data = domain.get("slots")

        main_intent = tracker.latest_message.get("intent").get("name")
        # 如果接收的是“/XX/XX”类似的意图消息，直接utter_/XX/XX
        if '/' in main_intent and '/其他' != main_intent[-3:]:
            dispatcher.utter_message(response=f"utter_{main_intent}")
            return [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for slot_name in clear_slots]

        catch_other_intent = False
        if '其他' in main_intent or 'nlu_fallback' == main_intent:  # 第二次nlu_fallback出现，会直接调用CQA
            catch_other_intent = True
            full_intent = main_intent
        else:
            full_intent = (
                tracker.latest_message.get("response_selector", {})
                    .get(main_intent, {})
                    .get("response", {})
                    .get("intent_response_key")
            )
            if "其他" in full_intent:
                catch_other_intent = True
        # 记录用户的输入
        user_query = tracker.latest_message.get("text")
        if user_query[0] == '/':
            user_query = tracker.get_slot('user_query')
        if catch_other_intent:
            message_title = (
                "您可能想问这些问题："
            )
            if "out_of_scope" in full_intent:
                button_title = ["我能问你什么问题呢", "你给我卖个萌吧", "你是谁", "你能给我点鼓励吗", "你给我讲个笑话吧"]
                button_payloads = ["/chitchat/ask_whatspossible", "/chitchat/卖个萌", "/chitchat/ask_whoisit",
                                   "/chitchat/鼓励", "/chitchat/讲个笑话"]
                buttons = []
                for title, payload in zip(button_title, button_payloads):
                    text = "{'out_of_scope':{'query': '%s'}}" % title
                    buttons.append({"title": text, "payload": payload})
                dispatcher.utter_message(text=message_title, buttons=buttons)
            else:
                second_sub_intent = {'confidence': 0}
                other_sub_intents = []
                try:
                    other_sub_intents = tracker.latest_message.get("response_selector", {}).get(main_intent, {}).get(
                        "ranking")[1:]
                    second_sub_intent = other_sub_intents[0]
                except Exception as e:
                    logger.warning(e)
                    logger.warning("can't get other sub intents such this intent come from fallback")
                # 只有第二个子意图的置信度大于0.8时，才推荐FAQ
                if second_sub_intent['confidence'] > 0.8:
                    buttons = []
                    for line in other_sub_intents[:5]:
                        intent = line['intent_response_key']
                        button_title = self.get_button_title(intent)
                        text = "{'faq':{'query': '%s'}}" % button_title
                        print("faq intent", intent)
                        buttons.append({"title": text, "payload": f"/{intent}"})
                    text = "{'faq':{'query': '%s'}}" % "以上都不是"
                    buttons.append({"title": text, "payload": "/deny"})

                    dispatcher.utter_message(text=message_title, buttons=buttons)
                # 否则，直接CQA
                else:
                    # TODO search in CQA
                    if not user_query:
                        # 如果用户之前没有问过问题，但触发了deny意图，直接回复抱歉
                        dispatcher.utter_message(template='utter_canthelp')
                        return [SlotSet('department', slots_data.get('department')['initial_value'])]
                    documents_ranked, scores_ranked = CQA_ES.query(topk=5, query=user_query, return_scores=True)
                    scores_ranked = sorted(scores_ranked.items(), key=lambda x: float(x[1]), reverse=True)
                    cqa_confidence = scores_ranked[0][1]
                    print('cqa', user_query)
                    print('cqa_confidence', cqa_confidence)
                    # TODO set threshold
                    threshold = 10
                    # 只有置信度大于阈值时，才推荐CQA
                    if cqa_confidence > threshold:
                        message_title = (
                            "为您找到这些相似问题："
                        )
                        buttons = []
                        for pid, _ in scores_ranked:
                            document = documents_ranked[pid]
                            title, query, answer = document.split('\t')
                            text = "{'cqa':{'query': '%s', 'answer': '%s'}}" % (query, answer)
                            buttons.append({"title": text, "payload": ''})
                        text = "{'cqa':{'query': '%s', 'answer': '%s'}}" % ("以上都不是", "")
                        buttons.append({"title": text, "payload": "/deny"})
                        dispatcher.utter_message(text=message_title, buttons=buttons)
                        return [SlotSet('user_query', user_query)] + [SlotSet('CQA_has_started', True)] + [
                            SlotSet('department', slots_data.get('department')['initial_value'])]
                    # 否则，直接DQA
                    else:
                        if not user_query:
                            # 如果用户之前没有问过问题，但触发了deny意图，直接回复抱歉
                            dispatcher.utter_message(template='utter_canthelp')
                            return [SlotSet('department', slots_data.get('department')['initial_value'])]
                        dqa_has_started = tracker.get_slot('DQA_has_started')
                        print('dqa', user_query)
                        # TODO search in DocumentQA
                        documents_ranked, scores_ranked = DQA_ES.query(topk=10, query=user_query,
                                                                       return_scores=True)
                        scores_ranked = _compute_softmax(scores_ranked)

                        input_datas = []
                        pid_set = []
                        for pid, _ in scores_ranked.items():
                            if pid not in pid_set:
                                pid_set.append(pid)
                            item = documents_ranked[pid]
                            document = item['document']
                            input_data = {'title': '',
                                          'document': document,
                                          'document_id': pid,
                                          'question': user_query}
                            input_datas.append(input_data)

                        print('input_datas', input_datas)

                        results = predict(model, tokenizer, input_datas)
                        print('results', results)

                        buttons = []
                        # 同一文档可能召回多个切片
                        had_seen_pid = []
                        for pid in scores_ranked.keys():
                            ans_dict = results[pid]
                            ans, qa_score = ans_dict['text'], ans_dict['score']
                            rank_score = scores_ranked[pid]
                            if ans == 'no answer' or qa_score == 0 or rank_score == 0:
                                continue
                            if qa_score > 0.7:
                                had_seen_pid.append(pid.split('seg')[0])
                                title, src = documents_ranked[pid]['title'], documents_ranked[pid]['src']
                                text = "{'dqa':{'ans': '%s', 'title': '%s', 'src': '%s'}}" % (ans, title, src)
                                buttons.append({"title": text, "payload": ''})
                                break
                        if buttons:
                            # 如果抽取出答案，给出答案以及来源，同时也呈现其他相关通知
                            for pid in pid_set:
                                if pid.split('seg')[0] not in had_seen_pid:
                                    had_seen_pid.append(pid.split('seg')[0])
                                    title, src = documents_ranked[pid]['title'], documents_ranked[pid]['src']
                                    text = "{'dqa':{'title': '%s', 'src': '%s'}}" % (title, src)
                                    buttons.append({"title": text, "payload": ''})
                            dispatcher.utter_message(text=' ', buttons=buttons)
                        else:
                            # 如果未抽取出答案，呈现相关通知
                            message_title = (
                                "为您在公文通中找到这些相关通知："
                            )
                            for pid in pid_set:
                                if pid.split('seg')[0] not in had_seen_pid:
                                    had_seen_pid.append(pid.split('seg')[0])
                                    title, src = documents_ranked[pid]['title'], documents_ranked[pid]['src']
                                    text = "{'dqa':{'title': '%s', 'src': '%s'}}" % (title, src)
                                    buttons.append({"title": text, "payload": ''})
                            dispatcher.utter_message(text=message_title, buttons=buttons)
                        return [SlotSet('user_query', user_query)] + [SlotSet('DQA_has_started', True)] + [
                            SlotSet('department', slots_data.get('department')['initial_value'])]
        else:
            dispatcher.utter_message(template=f"utter_{full_intent}")

        return [SlotSet('user_query', user_query)] + [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for
                                                      slot_name in clear_slots]

    def get_button_title(self, intent: Text) -> Text:

        utterances = self.intent_query_mappings.get(intent, 0)

        if utterances:
            button_title = utterances
        else:
            raise RuntimeError('没有找到意图对应的标准问题，请查看intent_description_mapping.csv文件！')

        return button_title


class ActionCommunityQA(Action):
    """Start the CommunityQA"""

    def name(self) -> Text:
        return "action_community_qa"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> List[EventType]:
        clear_slots = ['department']
        slots_data = domain.get("slots")
        user_query = tracker.get_slot('user_query')
        if not user_query:
            # 如果用户之前没有问过问题，但触发了deny意图，直接回复抱歉
            dispatcher.utter_message(template='utter_canthelp')
            return [SlotSet('department', slots_data.get('department')['initial_value'])]
        cqa_has_started = tracker.get_slot('CQA_has_started')
        if not cqa_has_started:
            print('cqa', user_query)
            # TODO search in CQA
            documents_ranked, scores_ranked = CQA_ES.query(topk=5, query=user_query, return_scores=True)
            scores_ranked = sorted(scores_ranked.items(), key=lambda x: float(x[1]), reverse=True)
            cqa_confidence = scores_ranked[0][1]
            print('cqa_confidence', cqa_confidence)
            # TODO set threshold
            threshold = 10
            # 只有置信度大于阈值时，才推荐CQA
            if cqa_confidence > threshold:
                message_title = (
                    "为您找到这些相似问题："
                )
                buttons = []
                for pid, _ in scores_ranked:
                    document = documents_ranked[pid]
                    title, query, answer = document.split('\t')
                    text = "{'cqa':{'query': '%s', 'answer': '%s'}}" % (query, answer)
                    buttons.append({"title": text, "payload": ''})
                text = "{'cqa':{'query': '%s', 'answer': '%s'}}" % ("以上都不是", "")
                buttons.append({"title": text, "payload": "/deny"})
                dispatcher.utter_message(text=message_title, buttons=buttons)
        return [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for slot_name in clear_slots]


class ActionDocumentQA(Action):
    """Start the DocumentQA"""

    def name(self) -> Text:
        return "action_document_qa"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> List[EventType]:
        clear_slots = ['department']
        slots_data = domain.get("slots")
        user_query = tracker.get_slot('user_query')
        if not user_query:
            # 如果用户之前没有问过问题，但触发了deny意图，直接回复抱歉
            dispatcher.utter_message(template='utter_canthelp')
            return [SlotSet('department', slots_data.get('department')['initial_value'])]
        dqa_has_started = tracker.get_slot('DQA_has_started')

        print('dqa', user_query)
        # TODO search in DocumentQA
        documents_ranked, scores_ranked = DQA_ES.query(topk=10, query=user_query, return_scores=True)
        scores_ranked = _compute_softmax(scores_ranked)

        input_datas = []
        pid_set = []
        for pid, _ in scores_ranked.items():
            if pid not in pid_set:
                pid_set.append(pid)
            item = documents_ranked[pid]
            document = item['document']
            input_data = {'title': '',
                          'document': document,
                          'document_id': pid,
                          'question': user_query}
            input_datas.append(input_data)

        print('input_datas', input_datas)

        results = predict(model, tokenizer, input_datas)
        print('results', results)

        buttons = []
        # 同一文档可能召回多个切片
        had_seen_pid = []
        for pid in scores_ranked.keys():
            ans_dict = results[pid]
            ans, qa_score = ans_dict['text'], ans_dict['score']
            rank_score = scores_ranked[pid]
            if ans == 'no answer' or qa_score == 0 or rank_score == 0:
                continue
            if qa_score > 0.7:
                had_seen_pid.append(pid.split('seg')[0])
                title, src = documents_ranked[pid]['title'], documents_ranked[pid]['src']
                text = "{'dqa':{'ans': '%s', 'title': '%s', 'src': '%s'}}" % (ans, title, src)
                buttons.append({"title": text, "payload": ''})
                break
        if buttons:
            # 如果抽取出答案，给出答案以及来源，同时也呈现其他相关通知
            for pid in pid_set:
                if pid.split('seg')[0] not in had_seen_pid:
                    had_seen_pid.append(pid.split('seg')[0])
                    title, src = documents_ranked[pid]['title'], documents_ranked[pid]['src']
                    text = "{'dqa':{'title': '%s', 'src': '%s'}}" % (title, src)
                    buttons.append({"title": text, "payload": ''})
            dispatcher.utter_message(text=' ', buttons=buttons)
        else:
            # 如果未抽取出答案，呈现相关通知
            message_title = (
                "为您在公文通中找到这些相关通知："
            )
            # 同一文档可能召回多个切片，只保留一个
            had_seen_pid = []
            for pid in pid_set:
                if pid.split('seg')[0] not in had_seen_pid:
                    had_seen_pid.append(pid.split('seg')[0])
                    title, src = documents_ranked[pid]['title'], documents_ranked[pid]['src']
                    text = "{'dqa':{'title': '%s', 'src': '%s'}}" % (title, src)
                    buttons.append({"title": text, "payload": ''})
            dispatcher.utter_message(text=message_title, buttons=buttons)
        return [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for slot_name in clear_slots]


class FindTheCorrespondingWEATHER(Action):
    def name(self) -> Text:
        return "FindTheCorrespondingweather"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:

        clear_slots = ['department', 'location', 'CQA_has_started', 'DQA_has_started']
        slots_data = domain.get("slots")

        user_in = tracker.latest_message.get("text")
        _, location_from_cpca = cpca.transform([user_in]).loc[0, ["省", "市"]]  # 提取text中的省和市

        location_from_slot = tracker.get_slot('location')
        location = location_from_cpca[:-1] if location_from_cpca else location_from_slot
        location = location if location else "深圳"
        timesss = '今天'
        times = ['今天', '明天', '后天', '大后天']
        for t in times:
            if t in user_in:
                timesss = t
                break

        d = date.today()
        if timesss == '今天':
            timesss = str(d)
        elif timesss == '明天':
            timesss = str(d + timedelta(days=1))
        elif timesss == '后天':
            timesss = str(d + timedelta(days=2))
        elif timesss == '大后天':
            timesss = str(d + timedelta(days=3))
        data, city = weather.weather_api(location)

        if city not in location:
            dispatcher.utter_message(template='utter_weather_exception')
            return [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for slot_name in clear_slots]

        try:
            text = ''
            for i in range(7):
                if data[i]['date'] == timesss:
                    index = ''.join([line['title'] + '：'
                                     + line['level'] + ' '
                                     + line['desc'] + '<br>' for line in data[i]['index']])
                    text = location + '天气：' \
                                      '<br>' + data[i]['date'] + ' ' + data[i]['week'] + \
                           '<br>天气状况：' + data[i]['wea'] + \
                           '<br>体感温度：' + data[i]['tem'] + \
                           '<br>最高温度：' + data[i]['tem1'] + \
                           '<br>最低温度：' + data[i]['tem2'] + \
                           '<br>湿度：' + data[i]['humidity'] + \
                           '<br>风向：' + data[i]['win'][0] + data[i]['win'][1] + \
                           '<br>风力等级：' + data[i]['win_speed'] + \
                           '<br>空气质量：' + data[i]['air_level'] + \
                           '<br>温馨提示：' + data[i]['air_tips'] + '<br>' + index

            dispatcher.utter_message(text="<div class='msg-text'>" + text + "</div>")

        except Exception as e:
            print('error', e)
            dispatcher.utter_message(template='utter_weather_exception')
        return [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for slot_name in clear_slots]


def _compute_softmax(scores):
    qid = list(scores.keys())
    scores = list(scores.values())

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return OrderedDict({i: j for i, j in zip(qid, probs)})
