import json
import os
from typing import List, Dict
import config


SAMPLE_MENU = [
    {
        "name": "Gỏi Ngó Sen Tôm Thịt",
        "short_description": "Ngó sen giòn, tôm sú, thịt ba chỉ, cà rốt, lạc rang, nước mắm chua ngọt.",
        "long_description": "Ngó sen tươi được sơ chế trắng giòn, trộn cùng tôm sú luộc, thịt ba chỉ thái sợi, cà rốt và rau răm. Rắc thêm đậu phộng rang và hành phi. Nước trộn gỏi chua ngọt cân bằng vị giác hoàn hảo.",
        "category": "Khai Vị & Món Cuốn",
        "price": 120000,
        "availability": True,
    },
    {
        "name": "Gỏi Bưởi Tôm Khô",
        "short_description": "Tép bưởi mọng nước, tôm khô một nắng, mực xé, chua ngọt thanh mát.",
        "long_description": "Sử dụng bưởi da xanh hoặc Năm Roi mọng nước, tách tép nguyên vẹn. Trộn cùng tôm khô một nắng loại ngon, mực khô xé sợi và nước sốt chua ngọt đặc biệt. Món ăn kích thích vị giác, thanh mát và giàu vitamin.",
        "category": "Khai Vị & Món Cuốn",
        "price": 135000,
        "availability": True,
    },
    {
        "name": "Bánh Xèo Miền Tây (Củ hũ dừa)",
        "short_description": "Vỏ bánh vàng giòn, nhân tôm thịt, củ hũ dừa, đậu xanh, ăn kèm rau rừng.",
        "long_description": "Chiếc bánh xèo size lớn, vỏ mỏng vàng ươm nhờ bột nghệ và nước cốt dừa, viền giòn rụm. Nhân bên trong gồm tôm đất, thịt ba chỉ, giá đỗ và đặc biệt là củ hũ dừa ngọt giòn. Ăn kèm rổ rau rừng (cải bẹ xanh, lá cóc, sao nhái).",
        "category": "Món Bánh",
        "price": 110000,
        "availability": True,
    },
    {
        "name": "Bánh Khọt Vũng Tàu (Tôm tươi)",
        "short_description": "Bánh tròn giòn rụm, nhân tôm tươi đỏ au, mỡ hành, bột tôm.",
        "long_description": "Những chiếc bánh tròn nhỏ được chiên giòn trong khuôn dầu, bên trên là tôm tươi nguyên con, rắc thêm bột tôm cháy và mỡ hành thơm phức. Cuốn cùng lá cải xanh và chấm nước mắm chua ngọt pha đu đủ bào.",
        "category": "Món Bánh",
        "price": 95000,
        "availability": True,
    },
    {
        "name": "Bánh Cuốn Thanh Trì Chả Quế",
        "short_description": "Bánh tráng mỏng tang, hành phi, chả quế thơm, nước chấm tinh dầu cà cuống.",
        "long_description": "Bánh cuốn tráng tay lớp mỏng tang, dai mềm, rắc hành phi vàng ruộm tự làm. Không nhân hoặc nhân thịt mộc nhĩ tuỳ chọn. Ăn kèm chả quế nướng thơm lừng và nước mắm chấm nhẹ dịu, có thể thêm tinh dầu cà cuống.",
        "category": "Món Bánh",
        "price": 85000,
        "availability": True,
    },
]


class MenuDataLoader:

    def __init__(self):
        self.menu_data = []
        self.load_menu()

    def load_menu(self):
        menu_file = os.path.join(config.DATA_DIR, "menu.json")

        if os.path.exists(menu_file):
            with open(menu_file, "r", encoding="utf-8") as f:
                self.menu_data = json.load(f)
        else:
            # Create sample menu if file doesn't exist
            self.menu_data = self.create_sample_menu()
            self.save_menu()

    def create_sample_menu(self) -> List[Dict]:
        return SAMPLE_MENU

    def save_menu(self):
        menu_file = os.path.join(config.DATA_DIR, "menu.json")
        with open(menu_file, "w", encoding="utf-8") as f:
            json.dump(self.menu_data, f, ensure_ascii=False, indent=2)

    def get_all_items(self) -> List[Dict]:
        return self.menu_data

    def get_item_by_name(self, name: str) -> Dict:
        name_lower = name.lower()
        for item in self.menu_data:
            if name_lower in item["name"].lower():
                return item
        return None

    def search_items(self, query: str) -> List[Dict]:
        query_lower = query.lower()
        results = []

        for item in self.menu_data:
            if (
                query_lower in item["name"].lower()
                or query_lower in item["description"].lower()
                or query_lower in item["category"].lower()
            ):
                results.append(item)

        return results

    def get_documents_for_rag(self) -> List[str]:
        documents = []

        for item in self.menu_data:
            doc = f"""\
Tên món ăn: {item['name']}
Món thuộc hạng mục: {item['category']}
Miêu tả ngắn: {item['short_description']}
Miêu tả dài: {item['long_description']}
Đơn giá: {item['price']:,}VNĐ
Trạng thái: {'Vẫn còn hàng' if item['availability'] else 'Đã hết hàng'} cho món {item['name']}"""
            documents.append(doc)

        return documents
    
    def save_documents(self) -> str:
        documents = self.get_documents_for_rag()
        documents = [f"[Món {i + 1}]\n{doc}" for i, doc in enumerate(documents)]
        content = "\n\n".join(documents)
        menu_file = os.path.join(config.DATA_DIR, "menu.txt")
        with open(menu_file, "w", encoding="utf-8") as file:
            file.write(content)
        return content


class InputLoader:

    @staticmethod
    def load_queries() -> List[str]:
        input_file = os.path.join(config.INPUT_DIR, "queries.txt")

        if not os.path.exists(input_file):
            # Create sample queries
            sample_queries = [
                "Có những món nào trong menu?",
                "Phở bò giá bao nhiêu?",
                "Có món gà rán không?",
                "Tôi muốn đặt 2 phần phở bò và 1 ly trà sữa ít đường",
                "Món nào giao nhanh nhất?",
                "Cho tôi xem các món bún",
                "Bún chả có những tùy chọn gì?",
                "Tôi muốn đặt bánh mì không ớt giao lúc 12 giờ",
                "Có đồ uống gì?",
                "Mì xào hải sản làm từ nguyên liệu gì?",
            ]

            os.makedirs(config.INPUT_DIR, exist_ok=True)
            with open(input_file, "w", encoding="utf-8") as f:
                f.write("\n".join(sample_queries))

            return sample_queries

        with open(input_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]

        return queries

    @staticmethod
    def load_answers() -> List[str]:
        answer_file = os.path.join(config.INPUT_DIR, "answers.txt")
        
        if not os.path.exists(answer_file):
            return []
        
        with open(answer_file, "r", encoding="utf-8") as f:
            answers = [line.strip() for line in f if line.strip()]
        
        return answers
