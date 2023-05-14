import collections
from app.base_model import DetectObject, DetectObjectWithRowColumn


class RecognitionRowColumn:
    def recognition(self, objects: list[DetectObject]) -> list[DetectObjectWithRowColumn]:
        recognition_objects = []
        for obj in objects:
            recognition_objects.append(DetectObjectWithRowColumn(score=obj.score, label=obj.label, box=obj.box, row=0, column=0))

        # 相同行的对象归类
        # [[obj1, obj2], [obj3, obj4, obj5]]
        obj_rows = []
        for obj in recognition_objects:
            if not obj_rows:
                obj_rows.append([obj])
                continue

            is_found = False
            for obj_row in obj_rows:
                if obj.box.is_same_row(obj_row[0].box):
                    obj_row.append(obj)
                    is_found = True
                    break

            if not is_found:
                obj_rows.append([obj])

        # 行排序。使用y坐标进行排序
        y_objrows = {}
        for obj_row in obj_rows:
            y_objrows[obj_row[0].box.y] = obj_row

        y_objrows = collections.OrderedDict(sorted(y_objrows.items()))

        for r, obj_row in enumerate(y_objrows.values()):
            # 列排序。使用x坐标进行行内的对象排序
            x_objs = {}
            for obj in obj_row:
                x_objs[obj.box.x] = obj

            x_objs = collections.OrderedDict(sorted(x_objs.items()))
            for c, obj in enumerate(x_objs.values()):
                obj.row = r + 1
                obj.column = c + 1

        # 纠正错位的对象，使其与正确的列对应。
        target_row_idx, target_row = self._get_row_with_most_objects(y_objrows.values())
        for r, obj_row in enumerate(y_objrows.values()):
            if r == target_row_idx:
                continue

            for obj in obj_row:
                for target_obj in target_row:
                    if obj.box.is_same_col(target_obj.box):
                        obj.column = target_obj.column
                        break

        return recognition_objects

    def _get_detect_object_with_row_column(self, obj: DetectObject) -> DetectObjectWithRowColumn:
        return DetectObjectWithRowColumn(score=obj.score, label=obj.label, box=obj.box, row=0, column=0)

    def _get_row_with_most_objects(self, objrows):
        idx, row, objs_num = 0, None, 0

        for r, obj_row in enumerate(objrows):
            num = len(obj_row)
            if num > objs_num:
                idx, row, objs_num = r, obj_row, num
                
        return idx, row

