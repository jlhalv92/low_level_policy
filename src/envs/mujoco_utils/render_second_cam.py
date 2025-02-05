import mujoco as mj
import numpy as np
from OpenGL.GL import *
import cv2


def get_renders(model, data, opt, scene, context, camera_name, loc_x, loc_y, width=640, height=480,
                       depth_flag=0, return_img=False):
    """
    Renders an inset view of the simulation scene from a different camera perspective.

    Parameters:
        model (mj.Model): The MuJoCo model.
        data (mj.Data): The MuJoCo simulation data.
        opt (mj.MjOption): The MuJoCo options.
        scene (mj.MjvScene): The MuJoCo scene.
        context (mj.MjrContext): The MuJoCo rendering context.
        camera_name (str): The name of the camera to use for the inset view.
        loc_x (int): The x-coordinate for the location of the inset view.
        loc_y (int): The y-coordinate for the location of the inset view.
        width (int): The width of the inset view.
        height (int): The height of the inset view.
        depth_flag (int): Flag to determine if depth information is rendered (0 for RGB, non-zero for depth).
    """

    height = int(height)
    width = int(width)


    offscreen_viewport = mj.MjrRect(int(loc_x), int(loc_y), int(width), int(height))

    # 2. Specify a different camera view by updating the scene with mjv_updateScene.
    # Set the camera to the specified view
    camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
    offscreen_cam = mj.MjvCamera()
    offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    offscreen_cam.fixedcamid = camera_id

    # Update scene for the off-screen camera
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)

    # 3. Render the scene in the offscreen buffer with mjr_render.
    mj.mjr_render(offscreen_viewport, scene, context)
    rgb = np.zeros((height * width * 3, 1), dtype=np.uint8)
    if not depth_flag:
        # buffer for rgb data


        # 4. Read the pixels with mjr_readPixels.
        mj.mjr_readPixels(rgb, None, offscreen_viewport, context)

        # 5. Call mjr_drawPixels using the rectangular viewport you created in step 1.
        if return_img:
            return np.flipud(rgb.reshape(height, width, 3))
        else:
            mj.mjr_drawPixels(rgb, None, offscreen_viewport, context)
    else:
        # buffer for depth data
        # depth: [numpy.ndarray[numpy.float32[m, 1]]], values are in range [0, 1]
        depth = np.zeros((height * width, 1), dtype=np.float32)
        depth8 = np.zeros((height, width, 3), dtype=np.uint8)

        # 4. Read the pixels with mjr_readPixels.
        mj.mjr_readPixels(rgb, depth, offscreen_viewport, context)


        depth_2d = depth.reshape(height, width)
        normalized_depth = cv2.normalize(depth_2d, None, 0, 255, cv2.NORM_MINMAX)
        # depth_image_8bit = cv2.flip(normalized_depth, 0)
        # cv2.imwrite('images/renderinstetwindowdepth.jpg', depth_image_8bit)

        # Convert normalized depth to 3-channel grayscale image
        depth8[:, :, 0] = depth8[:, :, 1] = depth8[:, :, 2] = normalized_depth

        # Flatten depth8 for drawing
        depth8 = depth8.flatten()

        # 5. Call mjr_drawPixels using the rectangular viewport you created in step 1.
        glClear(GL_DEPTH_BUFFER_BIT)  # allows rendering over geometries
        if return_img:
            return np.flipud(depth8.reshape(height, width, 3))

        else:
            mj.mjr_drawPixels(depth8, None, offscreen_viewport, context)

