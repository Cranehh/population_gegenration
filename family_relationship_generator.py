import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Dict, Tuple
import random

class FamilyRelationshipGenerator:
    """基于家庭特征生成家庭成员关系图的生成器"""
    
    def __init__(self):
        # 关系类型定义 (基于你的数据结构)
        self.relation_types = {
            0: '户主',
            1: '配偶', 
            2: '子女',
            3: '父母',
            4: '兄弟姐妹',
            5: '祖父母/外祖父母',
            6: '孙子女/外孙子女',
            7: '其他亲属',
            8: '非亲属'
        }
        
        # 年龄组定义
        self.age_groups = {
            'infant': (0, 3),      # 婴幼儿
            'child': (4, 12),      # 儿童
            'teen': (13, 17),      # 青少年
            'young_adult': (18, 35), # 青年
            'middle_aged': (36, 60), # 中年
            'elderly': (61, 120)     # 老年
        }
        
        # 家庭结构模板
        self.family_templates = {
            1: ['single_adult'],                    # 单人家庭
            2: ['couple', 'parent_child'],          # 两人家庭
            3: ['nuclear_family', 'three_generation'], # 三人家庭
            4: ['nuclear_family_4', 'extended_family'], # 四人家庭
            5: ['large_nuclear', 'extended_family'],    # 五人家庭
            6: ['large_extended', 'multi_generation'],  # 六人家庭
            7: ['large_extended', 'multi_generation'],  # 七人家庭
            8: ['large_extended', 'multi_generation']   # 八人家庭
        }
    
    def generate_family_structure(self, family_final_out: torch.Tensor) -> List[Dict]:
        """
        根据family_final_out生成家庭成员结构
        
        Args:
            family_final_out: [batch_size, feature_dim] 降噪后的家庭特征
                             第一维为家庭成员数量
        
        Returns:
            List[Dict]: 每个家庭的成员信息列表
        """
        batch_size = family_final_out.shape[0]
        families = []
        
        for i in range(batch_size):
            family_features = family_final_out[i]
            
            # 提取家庭特征
            family_info = self._parse_family_features(family_features)
            
            # 生成家庭成员
            family_members = self._generate_family_members(family_info)
            
            # 建立关系网络
            family_graph = self._build_family_graph(family_members, family_info)
            
            families.append({
                'family_id': f'family_{i:04d}',
                'family_info': family_info,
                'members': family_members,
                'relationships': family_graph
            })
        
        return families
    
    def _parse_family_features(self, family_features: torch.Tensor) -> Dict:
        """解析家庭特征"""
        # 假设family_final_out的结构：
        # [家庭成员数量, 工作人口数, 机动车数量, ..., have_student_prob, income_prob]
        
        family_size = int(torch.clamp(torch.round(family_features[0]), 1, 8).item())
        working_members = int(torch.clamp(torch.round(family_features[1]), 0, family_size).item())
        cars = int(torch.clamp(torch.round(family_features[2]), 0, 5).item())
        bikes = int(torch.clamp(torch.round(family_features[3]), 0, 10).item())
        e_bikes = int(torch.clamp(torch.round(family_features[4]), 0, 10).item())
        motorcycles = int(torch.clamp(torch.round(family_features[5]), 0, 5).item())
        elderly_cars = int(torch.clamp(torch.round(family_features[6]), 0, 5).item())
        
        # 解析概率分布
        have_student_prob = family_features[7:9].softmax(dim=0)
        have_student = torch.multinomial(have_student_prob, 1).item()
        
        income_prob = family_features[9:19].softmax(dim=0)
        income_level = torch.multinomial(income_prob, 1).item()
        
        return {
            'family_size': family_size,
            'working_members': working_members,
            'cars': cars,
            'bikes': bikes,
            'e_bikes': e_bikes,
            'motorcycles': motorcycles,
            'elderly_cars': elderly_cars,
            'have_student': bool(have_student),
            'income_level': income_level
        }
    
    def _generate_family_members(self, family_info: Dict) -> List[Dict]:
        """根据家庭信息生成家庭成员"""
        family_size = family_info['family_size']
        have_student = family_info['have_student']
        working_members = family_info['working_members']
        
        # 选择家庭结构模板
        template_options = self.family_templates.get(family_size, ['nuclear_family'])
        template = random.choice(template_options)
        
        # 根据模板生成成员
        members = self._create_members_by_template(template, family_info)
        
        # 确保成员数量正确
        while len(members) < family_size:
            members.append(self._create_additional_member(members, family_info))
        
        if len(members) > family_size:
            members = members[:family_size]
        
        # 分配工作状态
        self._assign_work_status(members, working_members)
        
        # 分配学生状态
        if have_student:
            self._assign_student_status(members)
        
        return members
    
    def _create_members_by_template(self, template: str, family_info: Dict) -> List[Dict]:
        """根据模板创建家庭成员"""
        members = []
        
        if template == 'single_adult':
            members.append(self._create_person('户主', age_range=(25, 65)))
            
        elif template == 'couple':
            members.append(self._create_person('户主', age_range=(25, 65)))
            members.append(self._create_person('配偶', age_range=(23, 63)))
            
        elif template == 'parent_child':
            if random.random() < 0.7:  # 70%概率是父母-子女
                members.append(self._create_person('户主', age_range=(35, 55)))
                members.append(self._create_person('子女', age_range=(5, 25)))
            else:  # 30%概率是成年子女-父母
                members.append(self._create_person('户主', age_range=(25, 45)))
                members.append(self._create_person('父母', age_range=(55, 75)))
                
        elif template == 'nuclear_family':
            members.append(self._create_person('户主', age_range=(30, 50)))
            members.append(self._create_person('配偶', age_range=(28, 48)))
            members.append(self._create_person('子女', age_range=(3, 20)))
            
        elif template == 'nuclear_family_4':
            members.append(self._create_person('户主', age_range=(32, 45)))
            members.append(self._create_person('配偶', age_range=(30, 43)))
            members.append(self._create_person('子女', age_range=(8, 18)))
            members.append(self._create_person('子女', age_range=(3, 15)))
            
        elif template == 'three_generation':
            members.append(self._create_person('户主', age_range=(30, 45)))
            members.append(self._create_person('配偶', age_range=(28, 43)))
            members.append(self._create_person('父母', age_range=(55, 70)))
            
        elif template in ['extended_family', 'large_nuclear', 'large_extended']:
            # 创建核心家庭
            members.append(self._create_person('户主', age_range=(30, 50)))
            members.append(self._create_person('配偶', age_range=(28, 48)))
            
            # 添加子女
            num_children = min(3, family_info['family_size'] - 2)
            for i in range(num_children):
                age_range = (2 + i*5, 18 + i*3)  # 不同年龄的子女
                members.append(self._create_person('子女', age_range=age_range))
            
        elif template == 'multi_generation':
            # 三代同堂
            members.append(self._create_person('户主', age_range=(35, 50)))
            members.append(self._create_person('配偶', age_range=(33, 48)))
            members.append(self._create_person('父母', age_range=(60, 75)))
            members.append(self._create_person('子女', age_range=(8, 20)))
            
            # 可能有孙辈
            if family_info['family_size'] > 4:
                members.append(self._create_person('孙子女/外孙子女', age_range=(1, 10)))
        
        return members
    
    def _create_person(self, relationship: str, age_range: Tuple[int, int] = (0, 80)) -> Dict:
        """创建单个家庭成员"""
        age = random.randint(age_range[0], age_range[1])
        gender = random.choice(['男', '女'])
        
        # 根据年龄和关系确定可能的属性
        education = self._determine_education(age, relationship)
        occupation = self._determine_occupation(age, relationship)
        has_license = self._determine_license(age)
        
        return {
            'relationship': relationship,
            'age': age,
            'gender': gender,
            'education': education,
            'occupation': occupation,
            'has_license': has_license,
            'is_working': False,  # 稍后分配
            'is_student': False   # 稍后分配
        }
    
    def _determine_education(self, age: int, relationship: str) -> str:
        """根据年龄和关系确定教育水平"""
        if age < 6:
            return '学前教育'
        elif age < 12:
            return '小学'
        elif age < 15:
            return '初中'
        elif age < 18:
            return '高中'
        elif age < 22:
            return random.choice(['高中', '大专', '本科'])
        elif age < 30:
            return random.choice(['大专', '本科', '硕士'])
        else:
            return random.choice(['初中', '高中', '大专', '本科', '硕士', '博士'])
    
    def _determine_occupation(self, age: int, relationship: str) -> str:
        """根据年龄和关系确定职业"""
        if age < 16:
            return '学生'
        elif age > 65:
            return '退休'
        else:
            occupations = [
                '企业管理人员', '专业技术人员', '办事人员', '商业服务业人员',
                '农林牧渔人员', '生产运输设备操作人员', '军人', '其他'
            ]
            return random.choice(occupations)
    
    def _determine_license(self, age: int) -> bool:
        """根据年龄确定是否有驾照"""
        if age < 18:
            return False
        elif age < 25:
            return random.random() < 0.3
        elif age < 60:
            return random.random() < 0.6
        else:
            return random.random() < 0.4
    
    def _create_additional_member(self, existing_members: List[Dict], family_info: Dict) -> Dict:
        """创建额外的家庭成员"""
        # 根据现有成员决定新成员的关系
        relationships = [m['relationship'] for m in existing_members]
        
        if '户主' not in relationships:
            return self._create_person('户主')
        elif '配偶' not in relationships and len(existing_members) > 1:
            return self._create_person('配偶')
        elif random.random() < 0.6:
            return self._create_person('子女', age_range=(1, 25))
        else:
            return self._create_person('其他亲属')
    
    def _assign_work_status(self, members: List[Dict], working_count: int):
        """分配工作状态"""
        # 找出可能工作的成员（18-65岁）
        workable_members = [m for m in members if 18 <= m['age'] <= 65]
        
        # 按年龄排序，优先给中年人分配工作
        workable_members.sort(key=lambda x: abs(x['age'] - 40))
        
        # 分配工作
        for i, member in enumerate(workable_members):
            if i < working_count:
                member['is_working'] = True
                if member['occupation'] == '学生':
                    member['occupation'] = '企业管理人员'  # 默认职业
    
    def _assign_student_status(self, members: List[Dict]):
        """分配学生状态"""
        # 找出可能是学生的成员
        student_candidates = [m for m in members if 3 <= m['age'] <= 25]
        
        if student_candidates:
            # 随机选择一个作为学生
            student = random.choice(student_candidates)
            student['is_student'] = True
            student['occupation'] = '学生'
    
    def _build_family_graph(self, members: List[Dict], family_info: Dict) -> nx.Graph:
        """构建家庭关系网络图"""
        G = nx.Graph()
        
        # 添加节点
        for i, member in enumerate(members):
            G.add_node(i, **member)
        
        # 添加关系边
        self._add_family_relationships(G, members)
        
        return G
    
    def _add_family_relationships(self, G: nx.Graph, members: List[Dict]):
        """添加家庭关系边"""
        # 找到户主
        head_idx = None
        spouse_idx = None
        children_idx = []
        parents_idx = []
        
        for i, member in enumerate(members):
            if member['relationship'] == '户主':
                head_idx = i
            elif member['relationship'] == '配偶':
                spouse_idx = i
            elif member['relationship'] == '子女':
                children_idx.append(i)
            elif member['relationship'] == '父母':
                parents_idx.append(i)
        
        # 添加夫妻关系
        if head_idx is not None and spouse_idx is not None:
            G.add_edge(head_idx, spouse_idx, relationship='夫妻')
        
        # 添加亲子关系
        if head_idx is not None:
            for child_idx in children_idx:
                G.add_edge(head_idx, child_idx, relationship='父子/母子')
            for parent_idx in parents_idx:
                G.add_edge(head_idx, parent_idx, relationship='父子/母子')
        
        if spouse_idx is not None:
            for child_idx in children_idx:
                G.add_edge(spouse_idx, child_idx, relationship='父子/母子')
        
        # 添加兄弟姐妹关系
        for i in range(len(children_idx)):
            for j in range(i+1, len(children_idx)):
                G.add_edge(children_idx[i], children_idx[j], relationship='兄弟姐妹')
    
    def visualize_family(self, family_data: Dict, save_path: str = None):
        """可视化家庭关系图"""
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False
        
        G = family_data['relationships']
        members = family_data['members']
        
        # 设置图形大小
        plt.figure(figsize=(12, 8))
        
        # 计算节点位置
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 根据关系类型设置节点颜色
        node_colors = []
        for member in members:
            if member['relationship'] == '户主':
                node_colors.append('red')
            elif member['relationship'] == '配偶':
                node_colors.append('pink')
            elif member['relationship'] == '子女':
                node_colors.append('lightblue')
            elif member['relationship'] == '父母':
                node_colors.append('orange')
            else:
                node_colors.append('lightgray')
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=1500, alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
        
        # 添加节点标签
        labels = {}
        for i, member in enumerate(members):
            labels[i] = f"{member['relationship']}\n{member['gender']}, {member['age']}岁"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        # 添加边标签
        edge_labels = nx.get_edge_attributes(G, 'relationship')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        # 设置标题
        family_info = family_data['family_info']
        title = f"家庭ID: {family_data['family_id']}\n"
        title += f"家庭规模: {family_info['family_size']}人, "
        title += f"工作人员: {family_info['working_members']}人, "
        title += f"有学生: {'是' if family_info['have_student'] else '否'}"
        
        plt.title(title, fontsize=12, pad=20)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_to_dataframe(self, families: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """导出为DataFrame格式"""
        family_records = []
        person_records = []
        
        for family in families:
            # 家庭记录
            family_info = family['family_info'].copy()
            family_info['家庭编号'] = family['family_id']
            family_records.append(family_info)
            
            # 个人记录
            for i, member in enumerate(family['members']):
                person_record = member.copy()
                person_record['家庭编号'] = family['family_id']
                person_record['成员编号'] = i
                person_records.append(person_record)
        
        family_df = pd.DataFrame(family_records)
        person_df = pd.DataFrame(person_records)
        
        return family_df, person_df


# 使用示例
def generate_families_from_dit_output(family_final_out: torch.Tensor):
    """从DiT输出生成家庭结构的主函数"""
    
    generator = FamilyRelationshipGenerator()
    
    # 生成家庭结构
    families = generator.generate_family_structure(family_final_out)
    
    # 导出为DataFrame
    family_df, person_df = generator.export_to_dataframe(families)
    
    print(f"成功生成 {len(families)} 个家庭")
    print(f"家庭DataFrame形状: {family_df.shape}")
    print(f"个人DataFrame形状: {person_df.shape}")
    
    # 可视化第一个家庭
    if len(families) > 0:
        generator.visualize_family(families[0])
    
    return families, family_df, person_df


if __name__ == "__main__":
    # 测试示例
    # 假设有5个家庭的DiT输出
    batch_size = 5
    feature_dim = 19  # 7个连续特征 + 2个学生特征 + 10个收入特征
    
    # 模拟family_final_out
    torch.manual_seed(42)
    family_final_out = torch.randn(batch_size, feature_dim)
    
    # 确保第一维（家庭规模）在合理范围内
    family_final_out[:, 0] = torch.clamp(torch.normal(3.5, 1.5, (batch_size,)), 1, 8)
    
    # 生成家庭
    families, family_df, person_df = generate_families_from_dit_output(family_final_out)
    
    print("\n家庭信息预览:")
    print(family_df.head())
    
    print("\n个人信息预览:")
    print(person_df.head())