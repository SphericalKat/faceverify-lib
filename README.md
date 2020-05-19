# Faceverify-lib
This project aims to provide a simple API for cropping faces from a bitmap and comparing two faces to verify that they're from the same person.

## Installation
In your project level `build.gradle` file, add the following:

```gradle
allprojects {
    repositories {
        google()
        jcenter()
        maven { url "http://jitpack.io/" }  // <-- THIS MUST BE ADDED
    }
}
```

In your app level `build.gradle` file, add the following dependency:
```gradle
dependencies {
	    implementation 'com.github.ATechnoHazard:faceverify-lib:0.1.0'
}
```

And finally, add this to the app-level `build.gradle` to ensure uncompressed models
```gradle
android {
    ...
    aaptOptions {
        noCompress "tflite"
    }
}
```

## Usage
### Initialize the library
```kotlin
lateinit var mtcnn: MTCNN
lateinit var mfn: MobileFaceNet
try {
    mtcnn = MTCNN(assets)
    mfn = MobileFaceNet(assets)
} catch (e: IOException) {
    Log.e("TAG", "Error initing models", e)
}
```

### Cropping the most prominent face from a bitmap
```kotlin
val croppedBitmap = FaceUtils.cropBitmapWithFace(someBitmap, mtcnn)
```
> **Note**: This might return a null bitmap if no faces were detected. Handle consequent errors appropriately.

### Comparing two bitmaps containing cropped faces
```kotlin
val similarity = mfn.compare(bitmap1, bitmap2)
if (similarity > MobileFaceNet.THRESHOLD) { // 0.8 by default, customize if you want
    // faces match
} else {
    // faces don't match
}
```

