import os
from datetime import datetime
from typing import Callable, Dict, List

import pandas as pd
from dotenv import load_dotenv
from slack_sdk.web import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse


class SlackFetcher:
    def __init__(self) -> None:
        load_dotenv()

        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.app_token = os.getenv("SLACK_APP_TOKEN")
        self.channel_id = os.getenv("SLACK_CHANNEL_ID")

        if not self.bot_token or not self.app_token or not self.channel_id:
            raise ValueError("Missing Slack environment variables")

        self.web_client = WebClient(token=self.bot_token)
        self.socket_client = SocketModeClient(
            app_token=self.app_token,
            web_client=self.web_client
        )

        self._user_cache: Dict[str, str] = {}

    def _get_username(self, user_id: str) -> str:
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        resp = self.web_client.users_info(user=user_id)
        name = resp["user"]["profile"].get("real_name") or resp["user"]["name"]
        self._user_cache[user_id] = name
        return name

    @staticmethod
    def _format_ts(ts: str) -> tuple[str, str]:
        dt = datetime.fromtimestamp(float(ts))
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")

    # -------- HISTORY --------
    def fetch_history(self, limit: int = 200) -> pd.DataFrame:
        messages: List[Dict[str, str]] = []
        cursor = None

        while True:
            resp = self.web_client.conversations_history(
                channel=self.channel_id,
                limit=limit,
                cursor=cursor
            )

            for msg in resp["messages"]:
                if msg.get("subtype"):
                    continue

                date, time_ = self._format_ts(msg["ts"])
                messages.append({
                    "date": date,
                    "time": time_,
                    "user_name": self._get_username(msg["user"]),
                    "message": msg.get("text", "")
                })

            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return pd.DataFrame(messages)

    # -------- LIVE --------
    def start_live_listener(self, on_message_callback: Callable[[Dict[str, str]], None]) -> None:
        def handler(client: SocketModeClient, req: SocketModeRequest) -> None:
            if req.type != "events_api":
                return

            client.send_socket_mode_response(
                SocketModeResponse(envelope_id=req.envelope_id)
            )

            event = req.payload["event"]
            if (
                event.get("type") == "message"
                and not event.get("subtype")
                and event.get("channel") == self.channel_id
            ):
                date, time_ = self._format_ts(event["ts"])
                on_message_callback({
                    "date": date,
                    "time": time_,
                    "user_name": self._get_username(event["user"]),
                    "message": event.get("text", "")
                })

        self.socket_client.socket_mode_request_listeners.append(handler)
        self.socket_client.connect()


def fetch_history(limit: int = 200) -> pd.DataFrame:
    """Convenience wrapper preserving legacy API."""
    return SlackFetcher().fetch_history(limit=limit)


def start_live_listener(on_message_callback: Callable[[Dict[str, str]], None]) -> None:
    """Convenience wrapper preserving legacy API."""
    SlackFetcher().start_live_listener(on_message_callback)
