# coding: utf-8
from abc import ABC, abstractmethod
from schemas.preprocess_result import PreprocessResult
from typing import List


class BasePreprocessService(ABC):
    @abstractmethod
    def call(self) -> PreprocessResult:
        pass

    @abstractmethod
    def ori_label_2_id_map(self, label: str) -> dict:
        pass
        

class PreprocessFactory():
    def create(self, service_name: str) -> BasePreprocessService:
        if service_name == "MAR20":
            from services.mars20_preprocess_service import Mars20PreprocessService
            return Mars20PreprocessService()
        elif service_name == "BridgeDataset":
            from services.bridge_preprocess_service import BridgePreprocessService
            return BridgePreprocessService()
        else:
            raise ValueError("Invalid service name")
