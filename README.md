# Robotron
An attempt at reinforcement learning using the computer to play the game Robotron on the XBox 360.

Setup:
I have an arduino hooked up to an X-Arcade joystick module for the XBox 360.  This allows me to send commands to the xbox via a serial connection to the Arduino.
I also have a HDMI capture card reading the video from the xbox.  I'm using this video to capture the score, lives, and playarea.  I use those to generate the reward and train the network.

If you have any advice or suggestions, please let me know!

You can watch it training (most of the time) at: https://www.twitch.tv/stridera
