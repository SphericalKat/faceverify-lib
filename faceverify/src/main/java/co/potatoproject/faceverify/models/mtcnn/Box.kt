package co.potatoproject.faceverify.models.mtcnn

import android.graphics.Point
import android.graphics.Rect
import kotlin.math.max

class Box {
    var box: IntArray = IntArray(4)
    var score // probability
            = 0f
    var bbr // bounding box regression
            : FloatArray = FloatArray(4)
    var deleted: Boolean = false
    var landmark // facial landmark.只有ONet输出Landmark
            : Array<Point?> = arrayOfNulls(5)

    fun left(): Int {
        return box[0]
    }

    fun right(): Int {
        return box[2]
    }

    fun top(): Int {
        return box[1]
    }

    fun bottom(): Int {
        return box[2]
    }

    fun width(): Int {
        return box[2] - box[0] + 1
    }

    fun height(): Int {
        return box[3] - box[1] + 1
    }

    fun transform2Rect(): Rect {
        val rect = Rect()
        rect.left = box[0]
        rect.top = box[1]
        rect.right = box[2]
        rect.bottom = box[3]
        return rect
    }

    fun area(): Int {
        return width() * height()
    }

    // Bounding Box Regression
    fun calibrate() {
        val w = box[2] - box[0] + 1
        val h = box[3] - box[1] + 1
        box[0] = (box[0] + w * bbr[0]).toInt()
        box[1] = (box[1] + h * bbr[1]).toInt()
        box[2] = (box[2] + w * bbr[2]).toInt()
        box[3] = (box[3] + h * bbr[3]).toInt()
        for (i in 0..3) bbr[i] = 0.0f
    }

    fun toSquareShape() {
        val w = width()
        val h = height()
        if (w > h) {
            box[1] -= (w - h) / 2
            box[3] += (w - h + 1) / 2
        } else {
            box[0] -= (h - w) / 2
            box[2] += (h - w + 1) / 2
        }
    }

    fun limitSquare(w: Int, h: Int) {
        if (box[0] < 0 || box[1] < 0) {
            val len = max(-box[0], -box[1])
            box[0] += len
            box[1] += len
        }
        if (box[2] >= w || box[3] >= h) {
            val len = max(box[2] - w + 1, box[3] - h + 1)
            box[2] -= len
            box[3] -= len
        }
    }

    fun transbound(w: Int, h: Int): Boolean {
        if (box[0] < 0 || box[1] < 0) {
            return true
        } else if (box[2] >= w || box[3] >= h) {
            return true
        }
        return false
    }

}