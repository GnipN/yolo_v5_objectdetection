import 'dart:math' as math;

class Box {
  final double x1, y1, x2, y2, score;
  Box(this.x1, this.y1, this.x2, this.y2, this.score);
}

double iou(Box a, Box b) {
  final areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  final areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

  final intersectionX1 = math.max(a.x1, b.x1);
  final intersectionY1 = math.max(a.y1, b.y1);
  final intersectionX2 = math.min(a.x2, b.x2);
  final intersectionY2 = math.min(a.y2, b.y2);

  final intersectionArea = math.max(0, intersectionX2 - intersectionX1) *
      math.max(0, intersectionY2 - intersectionY1);

  return intersectionArea / (areaA + areaB - intersectionArea);
}

List<Box> nonMaximumSuppressionYoLo(List<Box> boxes, double threshold) {
  boxes.sort((a, b) => b.score.compareTo(a.score));

  final selectedBoxes = <Box>[];
  while (boxes.isNotEmpty) {
    final box = boxes.removeAt(0);
    selectedBoxes.add(box);

    boxes.removeWhere((b) => iou(box, b) > threshold);
  }

  return selectedBoxes;
}
