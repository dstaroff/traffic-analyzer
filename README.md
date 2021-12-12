# Traffic analyzer

This piece of software allows detecting vehicles on video/camera source.
Setting measurement field with known distance makes it possible to estimate vehicle speed.
It also can make bruteforce analysis of traffic load in a way of dividing area occupied by vehicles by total measurement field area.

![Normal mode example](./gif/example_normal.gif)

## Debug mode

You can switch debug mode by pressing `Ctrl`+`D`.
In this mode, you can see the trace of each vehicle as well as its mask and a circle in which vehicle searches itself on a previous frame.

![Normal mode example](./gif/example_debug.gif)

## Credits

In this software we use a modified version of [Mask-RCNN](https://github.com/matterport/Mask_RCNN).
We have removed all the parts that needs for model training.
Also, the big part of utilitary functions has been removed as it's not needed.
Some functions had been moved or refactored to increase development and runtime efficiency.

## Warranties

This software is provided as-is and gives no warranties at all.
