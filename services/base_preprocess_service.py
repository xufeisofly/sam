# coding: utf-8
from abc import ABC, abstractmethod
from schemas.preprocess_result import PreprocessResult

class BasePreprocessService(ABC):
    def __init__(self, dataset_path=None) -> None:
        super().__init__()
    
    @abstractmethod
    def call(self, limit=-1) -> PreprocessResult:
        pass

    @abstractmethod
    def ori_label_2_id_map(self, label: str) -> dict:
        pass

class PreprocessFactory():
    def create(self, service_name: str, dataset_path=None) -> BasePreprocessService:
        if service_name == "MAR20":
            from services.mars20_preprocess_service import Mars20PreprocessService
            return Mars20PreprocessService(dataset_path=dataset_path)
        elif service_name == "BridgeDataset":
            from services.bridge_preprocess_service import BridgePreprocessService
            return BridgePreprocessService(dataset_path=dataset_path)
        elif service_name == "SODA-D":
            from services.dior_sodad_preprocess_service import DiorSodaDPreprocessService
            return DiorSodaDPreprocessService(dataset_path=dataset_path)
        elif service_name == "WHU-RS19":
            from services.whurs19_preprocess_service import Whurs19PreprocessService
            return Whurs19PreprocessService(dataset_path=dataset_path)
        elif service_name == "HRSC":
            from services.hrsc_preprocess_service import HRSCPreprocessService
            return HRSCPreprocessService(dataset_path=dataset_path)
        elif service_name == "SISI":
            from services.sisi_preprocess_service import SISIPreprocessService
            return SISIPreprocessService(dataset_path=dataset_path)
        else:
            raise ValueError("Invalid service name")
