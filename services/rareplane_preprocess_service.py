# coding: utf-8
import os
import json
import numpy as np
from lxml import etree
from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType

def geo_to_pixel(geo_transform, x_geo, y_geo):
        a, b, _, d, _, f = geo_transform
        col = (x_geo - a) / b
        row = (y_geo - d) / f
        return int(col), int(row)

class RarePlanesPreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "RarePlanes") if dataset_path is None else dataset_path
    
    def _get_data_paths(self, data_type: str):
        image_dir = {
            "train": "train_PS-RGB_tiled",
            "test": "test_PS-RGB_tiled"
        }[data_type]

        geojson_dir = {
            "train": "train_geojson_aircraft_tiled",
            "test": "test_geojson_aircraft_tiled"
        }[data_type]

        image_path = os.path.join(self._dataset_path, image_dir)
        geojson_path = os.path.join(self._dataset_path, geojson_dir)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image directory {image_path} does not exist.")
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"Annotation directory {geojson_path} does not exist.")

        return image_path, geojson_path

    def get_geo_transform(self, xml_file_path):
        tree = etree.parse(xml_file_path)
        root = tree.getroot()
        geo_transform_str = root.find('GeoTransform').text.strip().strip(',')
        geo_transform = [float(val) for val in geo_transform_str.split(',')]
        return geo_transform


    def process_data_type(self, data_type: str, limit=-1) -> PreprocessResult:
        image_path, geojson_path = self._get_data_paths(data_type)

        all_geojson_files = os.listdir(geojson_path)
        all_geojson_files = [f for f in all_geojson_files if f.endswith('.geojson')]

        result = PreprocessResult()

        for geojson_file in all_geojson_files[:limit] if limit > 0 else all_geojson_files:
            with open(os.path.join(geojson_path, geojson_file), 'r') as file:
                geojson_data = json.load(file)

            img_file_name = os.path.splitext(geojson_file)[0] + ".png"
            img_file_path = os.path.join(image_path, img_file_name)
            if not os.path.exists(img_file_path):
                print(f"Warning: PNG file {img_file_path} not found.")
                continue

            xml_file_name = f"{os.path.splitext(geojson_file)[0]}.png.aux.xml"
            xml_file_path = os.path.join(image_path, xml_file_name)
            if not os.path.exists(xml_file_path):
                print(f"Warning: XML file {xml_file_path} not found.")
                continue

            geo_transform = self.get_geo_transform(xml_file_path)

            result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=DataType[data_type.upper()])

            for feature in geojson_data['features']:
                if feature['geometry']['type'] == 'Polygon':
                    polygon_coords = feature['geometry']['coordinates'][0]
                    pixel_coords = [geo_to_pixel(geo_transform, x, y) for x, y in polygon_coords]
                    
                    cols, rows = zip(*pixel_coords)
                    min_col, max_col = min(cols), max(cols)
                    min_row, max_row = min(rows), max(rows)

                    box_array = np.array([int(min_col),
                                          int(min_row),
                                          int(max_col),
                                          int(max_row)])

                    box_item = BoxItem(
                        ori_label=self._get_ori_label_by_token(feature['properties']['role_id']),
                        box_array=np.array(box_array)
                    )
                    result_item.append(box_item)

            result.append(result_item)
        
        return result


    def call(self, limit=-1) -> dict:
        train_result = self.process_data_type("train",limit)
        test_result = self.process_data_type("test",limit)
        
        result = PreprocessResult(
            train_result_list=train_result.result_list,
            test_result_list=test_result.result_list)
        return result
        


    def ori_label_2_id_map(self) -> dict:
        return {
            "Small Civil Transport" : 255,
            "Medium Civil Transport" : 254,
            "Large Civil Transport" : 253,
            "Military Transport" : 252,
            "Military Bomber" : 251,
            "Military Fighter" : 250,
            "Military Trainer" : 249,
        }

    def _get_ori_label_by_token(self, label_token: int) -> str:
        token2ori = {
            1 : "Small Civil Transport",
            2 : "Medium Civil Transport",
            3 : "Large Civil Transport",
            4 : "Military Transport",
            5 : "Military Bomber",
            6 : "Military Fighter",
            7 : "Military Trainer"
        }
        return token2ori[label_token]