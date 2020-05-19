package co.potatoproject.faceverify.models.mtcnn

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Point
import co.potatoproject.faceverify.utils.FaceUtils
import org.tensorflow.lite.Interpreter
import java.util.*
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt


class MTCNN(assetManager: AssetManager?) {
    private val factor = 0.709f
    private val pNetThreshold = 0.6f
    private val rNetThreshold = 0.7f
    private val oNetThreshold = 0.7f
    private val pInterpreter: Interpreter
    private val rInterpreter: Interpreter
    private val oInterpreter: Interpreter

    fun detectFaces(bitmap: Bitmap, minFaceSize: Int): Vector<Box> {
        var boxes: Vector<Box>
        try {
            //【1】PNet generate candidate boxes
            boxes = pNet(bitmap, minFaceSize)
            squareLimit(boxes, bitmap.width, bitmap.height)

            //【2】RNet
            boxes = rNet(bitmap, boxes)
            squareLimit(boxes, bitmap.width, bitmap.height)

            //【3】ONet
            boxes = oNet(bitmap, boxes)
        } catch (e: IllegalArgumentException) {
            e.printStackTrace()
            boxes = Vector()
        }
        return boxes
    }

    private fun squareLimit(boxes: Vector<Box>, w: Int, h: Int) {
        // square
        for (i in boxes.indices) {
            boxes[i].toSquareShape()
            boxes[i].limitSquare(w, h)
        }
    }

    private fun pNet(bitmap: Bitmap, minSize: Int): Vector<Box> {
        val whMin = min(bitmap.width, bitmap.height)
        var currentFaceSize =
            minSize.toFloat() // currentFaceSize=minSize/(factor^k) k=0,1,2... until exceed whMin
        val totalBoxes = Vector<Box>()
        //【1】Image Pyramid and Feed to Pnet
        while (currentFaceSize <= whMin) {
            val scale = 12.0f / currentFaceSize

            // (1)Image Resize
            val bm: Bitmap = FaceUtils.bitmapResize(bitmap, scale)
            val w = bm.width
            val h = bm.height

            // (2)RUN CNN
            val outW = (ceil(w * 0.5 - 5) + 0.5).toInt()
            val outH = (ceil(h * 0.5 - 5) + 0.5).toInt()
            var prob1 =
                Array(
                    1
                ) {
                    Array(
                        outW
                    ) { Array(outH) { FloatArray(2) } }
                }
            var conv4_2_BiasAdd =
                Array(
                    1
                ) {
                    Array(
                        outW
                    ) { Array(outH) { FloatArray(4) } }
                }
            pNetForward(bm, prob1, conv4_2_BiasAdd)
            prob1 = FaceUtils.transposeBatch(prob1)
            conv4_2_BiasAdd = FaceUtils.transposeBatch(conv4_2_BiasAdd)

            val curBoxes = Vector<Box>()
            generateBoxes(prob1, conv4_2_BiasAdd, scale, curBoxes)

            // (4)nms 0.5
            nms(curBoxes, 0.5f, "Union")

            // (5)add to totalBoxes
            for (i in curBoxes.indices) if (!curBoxes[i].deleted) totalBoxes.addElement(
                curBoxes[i]
            )

            currentFaceSize /= factor
        }

        // NMS 0.7
        nms(totalBoxes, 0.7f, "Union")

        // BBR
        boundingBoxRegression(totalBoxes)
        return updateBoxes(totalBoxes)
    }

    private fun pNetForward(
        bitmap: Bitmap,
        prob1: Array<Array<Array<FloatArray>>>,
        conv4_2_BiasAdd: Array<Array<Array<FloatArray>>>
    ) {
        val img: Array<Array<FloatArray>> = FaceUtils.normalizeImage(bitmap)
        var pNetIn =
            Array(1) {
                Array(0) {
                    Array(0) {
                        FloatArray(0)
                    }
                }
            }
        pNetIn[0] = img
        pNetIn = FaceUtils.transposeBatch(pNetIn)
        val outputs: MutableMap<Int, Any> =
            HashMap()
        outputs[pInterpreter.getOutputIndex("pnet/prob1")] = prob1
        outputs[pInterpreter.getOutputIndex("pnet/conv4-2/BiasAdd")] = conv4_2_BiasAdd
        pInterpreter.runForMultipleInputsOutputs(arrayOf<Any>(pNetIn), outputs)
    }

    private fun generateBoxes(
        prob1: Array<Array<Array<FloatArray>>>,
        conv4_2_BiasAdd: Array<Array<Array<FloatArray>>>,
        scale: Float,
        boxes: Vector<Box>
    ): Int {
        val h: Int = prob1[0].size
        val w: Int = prob1[0][0].size
        for (y in 0 until h) {
            for (x in 0 until w) {
                val score = prob1[0][y][x][1]
                // only accept prob >threshold(0.6 here)
                if (score > pNetThreshold) {
                    val box = Box()
                    // core
                    box.score = score
                    // box
                    box.box[0] = (x * 2 / scale).roundToInt()
                    box.box[1] = (y * 2 / scale).roundToInt()
                    box.box[2] = ((x * 2 + 11) / scale).roundToInt()
                    box.box[3] = ((y * 2 + 11) / scale).roundToInt()
                    // bbr
                    for (i in 0..3) {
                        box.bbr[i] = conv4_2_BiasAdd[0][y][x][i]
                    }
                    // add
                    boxes.addElement(box)
                }
            }
        }
        return 0
    }

    private fun nms(
        boxes: Vector<Box>,
        threshold: Float,
        method: String
    ) {
        for (i in boxes.indices) {
            val box = boxes[i]
            if (!box.deleted) {
                for (j in i + 1 until boxes.size) {
                    val box2 = boxes[j]
                    if (!box2.deleted) {
                        val x1 = max(box.box[0], box2.box[0])
                        val y1 = max(box.box[1], box2.box[1])
                        val x2 = min(box.box[2], box2.box[2])
                        val y2 = min(box.box[3], box2.box[3])
                        if (x2 < x1 || y2 < y1) continue
                        val areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1)
                        var iou = 0f
                        if (method == "Union") iou =
                            1.0f * areaIoU / (box.area() + box2.area() - areaIoU) else if (method == "Min") iou =
                            1.0f * areaIoU / min(box.area(), box2.area())
                        if (iou >= threshold) { // 删除prob小的那个框
                            if (box.score > box2.score) box2.deleted = true else box.deleted = true
                        }
                    }
                }
            }
        }
    }

    private fun boundingBoxRegression(boxes: Vector<Box>) {
        for (i in boxes.indices) boxes[i].calibrate()
    }

    private fun rNet(bitmap: Bitmap, boxes: Vector<Box>): Vector<Box> {
        // RNet Input Init
        val num = boxes.size
        val rNetIn =
            Array(
                num
            ) {
                Array(
                    24
                ) { Array(24) { FloatArray(3) } }
            }
        for (i in 0 until num) {
            var curCrop: Array<Array<FloatArray>> =
                FaceUtils.cropAndResize(bitmap, boxes[i], 24)
            curCrop = FaceUtils.transposeImage(curCrop)
            rNetIn[i] = curCrop
        }

        // Run RNet
        rNetForward(rNetIn, boxes)

        // RNetThreshold
        for (i in 0 until num) {
            if (boxes[i].score < rNetThreshold) {
                boxes[i].deleted = true
            }
        }

        // Nms
        nms(boxes, 0.7f, "Union")
        boundingBoxRegression(boxes)
        return updateBoxes(boxes)
    }

    private fun rNetForward(
        rNetIn: Array<Array<Array<FloatArray>>>,
        boxes: Vector<Box>
    ) {
        val num = rNetIn.size
        val prob1 =
            Array(num) { FloatArray(2) }
        val conv5_2_conv5_2 =
            Array(num) { FloatArray(4) }
        val outputs: MutableMap<Int, Any> =
            HashMap()
        outputs[rInterpreter.getOutputIndex("rnet/prob1")] = prob1
        outputs[rInterpreter.getOutputIndex("rnet/conv5-2/conv5-2")] = conv5_2_conv5_2
        rInterpreter.runForMultipleInputsOutputs(arrayOf<Any>(rNetIn), outputs)

        for (i in 0 until num) {
            boxes[i].score = prob1[i][1]
            for (j in 0..3) {
                boxes[i].bbr[j] = conv5_2_conv5_2[i][j]
            }
        }
    }

    private fun oNet(bitmap: Bitmap, boxes: Vector<Box>): Vector<Box> {
        // ONet Input Init
        val num = boxes.size
        val oNetIn =
            Array(
                num
            ) {
                Array(
                    48
                ) { Array(48) { FloatArray(3) } }
            }
        for (i in 0 until num) {
            var curCrop: Array<Array<FloatArray>> =
                FaceUtils.cropAndResize(bitmap, boxes[i], 48)
            curCrop = FaceUtils.transposeImage(curCrop)
            oNetIn[i] = curCrop
        }

        // Run ONet
        oNetForward(oNetIn, boxes)
        // ONetThreshold
        for (i in 0 until num) {
            if (boxes[i].score < oNetThreshold) {
                boxes[i].deleted = true
            }
        }
        boundingBoxRegression(boxes)
        // Nms
        nms(boxes, 0.7f, "Min")
        return updateBoxes(boxes)
    }

    private fun oNetForward(
        oNetIn: Array<Array<Array<FloatArray>>>,
        boxes: Vector<Box>
    ) {
        val num = oNetIn.size
        val prob1 =
            Array(num) { FloatArray(2) }
        val conv6_2_conv6_2 =
            Array(num) { FloatArray(4) }
        val conv6_3_conv6_3 =
            Array(num) { FloatArray(10) }
        val outputs: MutableMap<Int, Any> =
            HashMap()
        outputs[oInterpreter.getOutputIndex("onet/prob1")] = prob1
        outputs[oInterpreter.getOutputIndex("onet/conv6-2/conv6-2")] = conv6_2_conv6_2
        outputs[oInterpreter.getOutputIndex("onet/conv6-3/conv6-3")] = conv6_3_conv6_3
        oInterpreter.runForMultipleInputsOutputs(arrayOf<Any>(oNetIn), outputs)

        for (i in 0 until num) {
            // prob
            boxes[i].score = prob1[i][1]
            // bias
            for (j in 0..3) {
                boxes[i].bbr[j] = conv6_2_conv6_2[i][j]
            }
            // landmark
            for (j in 0..4) {
                val x = (boxes[i].left() + conv6_3_conv6_3[i][j] * boxes[i].width()).roundToInt()
                val y =
                    (boxes[i].top() + conv6_3_conv6_3[i][j + 5] * boxes[i].height()).roundToInt()
                boxes[i].landmark[j] = Point(x, y)
            }
        }
    }

    companion object {
        private const val MODEL_FILE_PNET = "pnet.tflite"
        private const val MODEL_FILE_RNET = "rnet.tflite"
        private const val MODEL_FILE_ONET = "onet.tflite"

        fun updateBoxes(boxes: Vector<Box>): Vector<Box> {
            val b = Vector<Box>()
            for (i in boxes.indices) {
                if (!boxes[i].deleted) {
                    b.addElement(boxes[i])
                }
            }
            return b
        }
    }

    init {
        val options =
            Interpreter.Options()
        options.setNumThreads(4)
        pInterpreter = Interpreter(
            FaceUtils.loadModelFile(
                assetManager!!,
                MODEL_FILE_PNET
            ), options
        )
        rInterpreter = Interpreter(
            FaceUtils.loadModelFile(
                assetManager,
                MODEL_FILE_RNET
            ), options
        )
        oInterpreter = Interpreter(
            FaceUtils.loadModelFile(
                assetManager,
                MODEL_FILE_ONET
            ), options
        )
    }
}