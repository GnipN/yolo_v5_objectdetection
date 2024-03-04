/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import 'dart:developer';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'removeOverlapping.dart';

class ObjectDetection {
  // static const String _modelPath = 'assets/models/yolov2_tiny.tflite';   // output error
  // static const String _labelPath = 'assets/models/yolov2_tiny.txt';

  // static const String _modelPath =
  //     'assets/models/ssd_mobilenet.tflite'; // success
  // static const String _labelPath = 'assets/models/ssd_mobilenet.txt';

  static const String _modelPath =
      'assets/models/yolov5_dog.tflite'; // can process but not detect
  static const String _labelPath = 'assets/models/yolov5_dog.txt';

  // static const String _modelPath = 'assets/models/yolov5s-fp16.tflite';
  // static const String _labelPath = 'assets/models/yolov5s-fp16.txt';

  Interpreter? _interpreter;
  List<String>? _labels;

  ObjectDetection() {
    _loadModel();
    _loadLabels();
    log('Done.');
  }

  Future<void> _loadModel() async {
    log('Loading interpreter options...');
    final interpreterOptions = InterpreterOptions();

    // Use XNNPACK Delegate
    if (Platform.isAndroid) {
      interpreterOptions.addDelegate(XNNPackDelegate());
    }

    // Use Metal Delegate
    if (Platform.isIOS) {
      interpreterOptions.addDelegate(GpuDelegate());
    }

    log('Loading interpreter...');
    _interpreter =
        await Interpreter.fromAsset(_modelPath, options: interpreterOptions);
  }

  Future<void> _loadLabels() async {
    log('Loading labels...');
    final labelsRaw = await rootBundle.loadString(_labelPath);
    _labels = labelsRaw.split('\n');
  }

  Future<Uint8List> loadImageData(String path) async {
    ByteData data = await rootBundle.load(path);
    return data.buffer.asUint8List();
  }

  List<List<double>> nonMaximumSuppression(
      List<List<double>> boxes, double threshold) {
    // Sort the boxes by the objectness score in descending order
    boxes.sort((a, b) => b[4].compareTo(a[4]));

    List<List<double>> selectedBoxes = [];

    while (boxes.isNotEmpty) {
      // Select the box with the highest score
      List<double> box = boxes.removeAt(0);
      selectedBoxes.add(box);

      // Compare this box with the rest of the boxes
      boxes.removeWhere((otherBox) {
        // Calculate the Intersection over Union (IoU)
        double x1 = math.max(box[0], otherBox[0]);
        double y1 = math.max(box[1], otherBox[1]);
        double x2 = math.min(box[2], otherBox[2]);
        double y2 = math.min(box[3], otherBox[3]);

        double intersection = math.max(0, x2 - x1) * math.max(0, y2 - y1);
        double union = (box[2] - box[0]) * (box[3] - box[1]) +
            (otherBox[2] - otherBox[0]) * (otherBox[3] - otherBox[1]) -
            intersection;

        double iou = intersection / union;

        // Remove the box if the overlap is greater than the threshold
        return iou > threshold;
      });
    }

    return selectedBoxes;
  }

  Uint8List analyseImage(String imagePath) {
    log('Analysing image...');
    // Reading image bytes from file
    final imageData = File(imagePath).readAsBytesSync();
    // final imageData = File(imagePath).readAsBytesSync();
    // Uint8List imageData = await getBytesFromAsset('assets/images/my_image.png');

    // Decoding image
    final image = img.decodeImage(imageData);

    // Resizing image fpr model, [300, 300]
    final imageInput = img.copyResize(
      image!,
      width: 640,
      height: 640,
      // width: 300,
      // height: 300,
    );

    // Creating matrix representation, [300, 300, 3]
    final imageMatrix = List.generate(
      imageInput.height,
      (y) => List.generate(
        imageInput.width,
        (x) {
          final pixel = imageInput.getPixel(x, y);
          return [pixel.r, pixel.g, pixel.b];
        },
      ),
    );

    final output = _runInference(imageMatrix);

    // drawDetections(output, imageMatrix)

    log('Processing outputs...');
    // Location

    //  Apply a threshold to the objectness score
    // Float32List processedTensor = output.reshape([25200, 6]);
    List<Box> boxesYolo = [];
    int count = 0;
    if (kDebugMode) {
      print("Output ${output.length}");
    }
    for (var element in output) {
      if (kDebugMode) {
        print('element ${element.length}');
      }
      for (var elementD in element) {
        List<double> doubleList = (elementD as List).cast<double>();
        if (kDebugMode) {
          count++;

          if (kDebugMode) {
            if (elementD[4] > 0.8 && elementD[5] > 0.8) {
              // elementD[0] is x
              // elementD[1] is y
              // elementD[2] is width
              // elementD[3] is height
              // elementD[4] is objectness score
              // elementD[5] is 1 class probability

              // if (elementD[1] > 1.0) {
              print("$count elementD $elementD");
              print("\n");

              int imageWidth = 640;
              int imageHeight = 640;

              // draw rectangle to image for show result

              double x = elementD[0];
              double y = elementD[1];
              double width = elementD[2];
              double height = elementD[3];

              int x1 = ((x - width / 2) * imageWidth).toInt();
              int y1 = ((y - height / 2) * imageHeight).toInt();
              int x2 = ((x + width / 2) * imageWidth).toInt();
              int y2 = ((y + height / 2) * imageHeight).toInt();

              boxesYolo.add(Box(x1.toDouble(), y1.toDouble(), x2.toDouble(),
                  y2.toDouble(), elementD[4]));

              // draw text
              // img.drawString(
              //   imageInput,
              //   'Dog ${elementD[5]}',
              //   font: img.arial14,
              //   x: x1 + 1,
              //   y: y1 + 1,
              //   color: img.ColorRgb8(255, 0, 0),
              // );
            }
          }
          // if (count > 2) break;
          // print("rumtime type ${elementD.runtimeType}");
        }
        // element.forEach((element) {
        //   print(element);
        // });
      }
    }

    List<Box> selectedBoxesYolo = nonMaximumSuppressionYoLo(boxesYolo, 0.2);
    for (var box in selectedBoxesYolo) {
      // Do something with box
      img.drawRect(
        imageInput,
        x1: box.x1.toInt(),
        y1: box.y1.toInt(),
        x2: box.x2.toInt(),
        y2: box.y2.toInt(),
        color: img.ColorRgb8(255, 0, 0),
        thickness: 3,
      );

      // img.drawString(
      //           imageInput,
      //           'Dog ${elementD[5]}',
      //           font: img.arial14,
      //           x: x1 + 1,
      //           y: y1 + 1,
      //           color: img.ColorRgb8(255, 0, 0),
      //         );
    }

    log('Outlining objects...');
    // Label drawing

    //     // Label drawing
    //     img.drawString(
    //       imageInput,
    //       '${classication[i]} ${scores[i]}',
    //       font: img.arial14,
    //       x: locations[i][1] + 1,
    //       y: locations[i][0] + 1,
    //       color: img.ColorRgb8(255, 0, 0),
    //     );
    //   }
    // }

    log('Done.');

    return img.encodeJpg(imageInput);
  }

  List<List<Object>> _runInference(
    List<List<List<num>>> imageMatrix,
  ) {
    log('Running inference...');

    // Set input tensor [1, 300, 300, 3]
    final input = [imageMatrix];

    // ssd_mobilenet.tflite
    var output = List.filled(
        1, List.filled(25200, List.filled(6, 0.0))); // Adjust the shape here

    // _interpreter!.runForMultipleInputs([input], output);
    _interpreter!.run(input, output);
    return output.toList();
  }
}



// -------------- original code ------------------
// git clone https://github.com/tensorflow/flutter-tflite.git  



// The output shape of [1, 25200, 85] from a YOLOv5 model represents the following:

// 1: This is the batch size. It means the model is processing one image at a time.

// 25200: This is the total number of bounding boxes predicted by the model. YOLOv5 divides the input image into a grid, 
//and for each grid cell, it predicts multiple bounding boxes. 
//The number 25200 is the total number of grid cells times the number of bounding boxes per grid cell. 
//The exact number can vary depending on the specific configuration of the model.

// 85: This is the number of attributes for each bounding box. In YOLOv5, each bounding box has the following attributes:

// 4 values for the bounding box coordinates (x, y, width, height)
// 1 value for the objectness score, which indicates the probability that an object is present in the bounding box
// 80 values for the class probabilities, assuming the model was trained on a dataset with 80 classes (like the COCO dataset). 
//Each value represents the probability that the detected object belongs to a particular class.
// So, the output tensor of shape [1, 25200, 85] contains the bounding box coordinates, objectness scores, and 
//class probabilities for all predicted bounding boxes for a single image.




// This comment block is describing the expected shapes of the output tensors for a TensorFlow Lite object detection model. Each line corresponds to a different output tensor:
// Locations: [1, 10, 4]: This tensor contains the bounding box coordinates for each detected object. The shape [1, 10, 4] indicates that there is one batch, up to 10 detections per batch, and 4 coordinates for each detection (typically [ymin, xmin, ymax, xmax]).
// Classes: [1, 10]: This tensor contains the class labels for each detected object. The shape [1, 10] indicates that there is one batch and up to 10 detections per batch.
// Scores: [1, 10]: This tensor contains the confidence scores for each detected object. The shape [1, 10] indicates that there is one batch and up to 10 detections per batch.
// Number of detections: [1]: This tensor contains the total number of detections. The shape [1] indicates that there is one batch.
// These shapes are typical for a TensorFlow Lite object detection model that has been trained to detect up to 10 objects in an image. The exact shapes can vary depending on the specific model and how it was trained.





// usage: export.py [-h] [--data DATA] [--weights WEIGHTS [WEIGHTS ...]] [--imgsz IMGSZ [IMGSZ ...]]
//                  [--batch-size BATCH_SIZE] [--device DEVICE] [--half] [--inplace] [--keras]
//                  [--optimize] [--int8] [--per-tensor] [--dynamic] [--simplify] [--opset OPSET]
//                  [--verbose] [--workspace WORKSPACE] [--nms] [--agnostic-nms]
//                  [--topk-per-class TOPK_PER_CLASS] [--topk-all TOPK_ALL] [--iou-thres IOU_THRES]
//                  [--conf-thres CONF_THRES] [--include INCLUDE [INCLUDE ...]]

// options:
//   -h, --help            show this help message and exit
//   --data DATA           dataset.yaml path
//   --weights WEIGHTS [WEIGHTS ...]
//                         model.pt path(s)
//   --imgsz IMGSZ [IMGSZ ...], --img IMGSZ [IMGSZ ...], --img-size IMGSZ [IMGSZ ...]
//                         image (h, w)
//   --batch-size BATCH_SIZE
//                         batch size
//   --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
//   --half                FP16 half-precision export
//   --inplace             set YOLOv5 Detect() inplace=True
//   --keras               TF: use Keras
//   --optimize            TorchScript: optimize for mobile
//   --int8                CoreML/TF/OpenVINO INT8 quantization
//   --per-tensor          TF per-tensor quantization
//   --dynamic             ONNX/TF/TensorRT: dynamic axes
//   --simplify            ONNX: simplify model
//   --opset OPSET         ONNX: opset version
//   --verbose             TensorRT: verbose log
//   --workspace WORKSPACE
//                         TensorRT: workspace size (GB)
//   --nms                 TF: add NMS to model
//   --agnostic-nms        TF: add agnostic NMS to model
//   --topk-per-class TOPK_PER_CLASS
//                         TF.js NMS: topk per class to keep
//   --topk-all TOPK_ALL   TF.js NMS: topk for all classes to keep
//   --iou-thres IOU_THRES
//                         TF.js NMS: IoU threshold
//   --conf-thres CONF_THRES
//                         TF.js NMS: confidence threshold
//   --include INCLUDE [INCLUDE ...]
//                         torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite,
//                         edgetpu, tfjs, paddle

// !python export.py --weights yolov5s.pt --include tflite  --img 640 --nms --topk-per-class 1 --topk-all 10 --optimize --conf-thres 0.6