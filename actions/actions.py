# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

import mimetypes
import random
from typing import Any, Dict, List, Text, Optional

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher

import cpca  # cpca是chinese_province_city_area_mapper的简称，可用于处理中文地址
import validators

from actions.utils.coins import CoinDataManager
from actions.utils.request import get
from actions.utils.search import search_anime, AnimalImgSearch

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
from datetime import datetime, date, timedelta

USER_INTENT_OUT_OF_SCOPE = "out_of_scope"

logger = logging.getLogger(__name__)

INTENT_DESCRIPTION_MAPPING_PATH = "actions/intent_description_mapping.csv"

QUERY_KEY = "82510add8a7340caa9afcabfab78a639"

CITY_LOOKUP_URL = "https://geoapi.qweather.com/v2/city/lookup"
WEATHER_URL = "https://devapi.qweather.com/v7/weather/now"


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


class ActionAskWhoForContact(Action):
    """询问需要哪位老师或部门的联系方式"""

    def name(self) -> Text:
        return "action_ask_who_for_contact"

    def __init__(self) -> None:
        self.flag = True

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:

        clear_slots = ['department', 'full_name']
        slots_data = domain.get("slots")

        had_asked = check_last_event(tracker, "action_ask_who_for_contact", slots_data=slots_data)
        if had_asked and self.flag:
            dispatcher.utter_message(text="未能理解您的意思。")
            dispatcher.utter_message(template="utter_anything_else")
            self.flag = False
        else:
            dispatcher.utter_message(text="需要哪位老师或者哪个部门的联系方式呢？")
            self.flag = True

        return [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for slot_name in clear_slots]


class ActionQueryContactInformation(Action):
    """查询联系方式"""

    def name(self) -> Text:
        return "action_query_contact_information"

    @staticmethod
    def db() -> bool:
        """模拟数据库"""

        return True

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:

        department = tracker.get_slot("department")
        full_name = tracker.get_slot("full_name")
        intent = tracker.latest_message["intent"].get("name")
        clear_slots = ['department', 'full_name']
        slots_data = domain.get("slots")

        if '联系方式' in intent:

            if department == '未知部门' and full_name == '未指定具体老师':
                return [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for slot_name in clear_slots]
            else:

                if department != '未知部门' and full_name != '未指定具体老师':
                    dispatcher.utter_message(template="utter_search_contact_information1")
                elif department == '未知部门' and full_name != '未指定具体老师':
                    dispatcher.utter_message(template="utter_search_contact_information2")
                else:
                    dispatcher.utter_message(template="utter_search_contact_information3")

                self.db()
                dispatcher.utter_message(template="utter_search_failure")

        return [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for slot_name in clear_slots]


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

        clear_slots = ['department', 'full_name']
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
            if intent.get("name", "") not in ["chitchat", '图书馆服务', '就业指导', '后勤服务', '教学教务',
                                              '校园卡']
            else tracker.latest_message.get("response_selector")
                .get(intent.get("name", ""))
                .get("ranking")[0]
                .get("intent_response_key")
            for intent in intent_ranking
        ]
        if "nlu_fallback" in first_intent_names:
            first_intent_names.remove("nlu_fallback")
        if "/out_of_scope" in first_intent_names:
            first_intent_names.remove("/out_of_scope")
        if "out_of_scope" in first_intent_names:
            first_intent_names.remove("out_of_scope")

        if len(first_intent_names) > 0:
            message_title = (
                "对不起，我不太理解您的意思🤔，您是想说..."
            )

            entities = tracker.latest_message.get("entities", [])
            entities = {e["entity"]: e["value"] for e in entities}

            entities_json = json.dumps(entities)

            buttons = []
            for intent in first_intent_names:
                button_title = self.get_button_title(intent, entities)
                if len(entities_json) > 2:

                    buttons.append(
                        {"title": button_title, "payload": f"/{intent}{entities_json}"}
                    )
                else:
                    buttons.append({"title": button_title, "payload": button_title})

            buttons.append({"title": "都不是", "payload": "都不是"})

            dispatcher.utter_message(text=message_title, buttons=buttons)
        else:
            message_title = (
                "对不起，我不太理解您的意思"
                " 🤔 您可以问得再具体一些吗？"
            )
            dispatcher.utter_message(text=message_title)

        return [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for slot_name in clear_slots]

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

        # Fallback caused by TwoStageFallbackPolicy
        last_intent = tracker.latest_message["intent"]["name"]
        if last_intent in ["nlu_fallback", USER_INTENT_OUT_OF_SCOPE]:
            return [SlotSet("feedback_value", "negative")]

        # Fallback caused by Core
        else:
            dispatcher.utter_message(template="utter_canthelp")
            return [UserUtteranceReverted()]


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
        clear_slots = ['department', 'full_name']
        slots_data = domain.get("slots")

        main_intent = tracker.latest_message.get("intent").get("name")
        full_intent = (
            tracker.latest_message.get("response_selector", {})
                .get(main_intent, {})
                .get("response", {})
                .get("intent_response_key")
        )
        if "其他" in full_intent:
            message_title = (
                "您可能想问这些问题："
            )
            if "out_of_scope" in full_intent:
                button_title = ["我能问你什么问题呢", "你给我卖个萌吧", "你是谁", "你能给我点鼓励吗", "你给我讲个笑话吧"]
                buttons = []
                for title in button_title:
                    buttons.append({"title": title, "payload": title})
                dispatcher.utter_message(text=message_title, buttons=buttons)
            else:

                other_sub_intents = tracker.latest_message.get("response_selector", {}).get(main_intent, {}).get(
                    "ranking")[1:]

                buttons = []
                for line in other_sub_intents[:5]:
                    intent = line['intent_response_key']
                    button_title = self.get_button_title(intent)
                    buttons.append({"title": button_title, "payload": button_title})

                dispatcher.utter_message(text=message_title, buttons=buttons)
        else:
            dispatcher.utter_message(template=f"utter_{full_intent}")

        return [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for slot_name in clear_slots]

    def get_button_title(self, intent: Text) -> Text:

        utterances = self.intent_query_mappings.get(intent, 0)

        if utterances:
            button_title = utterances
        else:
            raise RuntimeError('没有找到意图对应的标准问题，请查看intent_description_mapping.csv文件！')

        return button_title


class ActionTagFeedback(Action):
    """Tag a conversation in Rasa X as positive or negative feedback """

    def name(self):
        return "action_tag_feedback"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:

        feedback = tracker.get_slot("feedback_value")
        slots_data = domain.get("slots")

        if feedback == "positive":
            label = '[{"value":"postive feedback","color":"76af3d"}]'
            dispatcher.utter_message(template='utter_pos_feedback')
        elif feedback == "negative":
            label = '[{"value":"negative feedback","color":"ff0000"}]'
            dispatcher.utter_message(template='utter_neg_feedback')

        # rasax = RasaXAPI()
        # rasax.tag_convo(tracker, label)

        return [SlotSet("feedback_value", slots_data.get("feedback_value")['initial_value'])]


class FindTheCorrespondingWEATHER(Action):
    def name(self) -> Text:
        return "FindTheCorrespondingweather"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:

        clear_slots = ['department', 'full_name', 'location']
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
                                     + line['desc'] + '\n' for line in data[i]['index']])
                    text = location + '天气：' \
                                      '\n' + data[i]['date'] + ' ' + data[i]['week'] + \
                           '\n天气状况：' + data[i]['wea'] + \
                           '\n体感温度：' + data[i]['tem'] + \
                           '\n最高温度：' + data[i]['tem1'] + \
                           '\n最低温度：' + data[i]['tem2'] + \
                           '\n湿度：' + data[i]['humidity'] + \
                           '\n风向：' + data[i]['win'][0] + data[i]['win'][1] + \
                           '\n风力等级：' + data[i]['win_speed'] + \
                           '\n空气质量：' + data[i]['air_level'] + \
                           '\n温馨提示：' + data[i]['air_tips'] + '\n' + index

            dispatcher.utter_message(text=text)

        except Exception as e:
            print('error', e)
            dispatcher.utter_message(template='utter_weather_exception')
        return [SlotSet(slot_name, slots_data.get(slot_name)['initial_value']) for slot_name in clear_slots]
