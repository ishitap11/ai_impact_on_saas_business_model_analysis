import pandas as pd


SIGNAL_VARS = ["Quarter", "PCS", "AMS", "MAS", "AES", "AACI Score", "Post-AI Event"]

class TeamviewerAIEvent:
    events_df = None

    def __init__(self):
        self.events_df = self._import_csv()

    def _import_csv(self):
        full_df = pd.read_csv('TeamViewer_AI_Event_Timeline.csv', sep=';')
        df = full_df[SIGNAL_VARS].copy()
        return df

    def get_ai_event_signal(self) -> pd.Series:
        return self.events_df["Post-AI Event"]

    def get_aaci_score(self) -> pd.Series:
        return self.events_df["AACI Score"]

    def get_complete_ai_timeline(self) -> pd.DataFrame:
        return self.events_df

