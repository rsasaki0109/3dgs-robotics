import { WebSocket } from 'ws';
import {
  normalizeRobotBridgeMessage,
  normalizeRobotBridgeUrl,
  parseRobotBridgeMessage,
  robotBridgeDefaultUrl,
  robotBridgeProtocolId,
  robotBridgeRosbridgeSource,
  stringifyRobotBridgeMessage
} from '../src/robotics-bridge.js';

const defaults = {
  roboticsUrl: robotBridgeDefaultUrl,
  rosbridgeUrl: process.env.DREAMWALKER_ROSBRIDGE_URL || 'ws://127.0.0.1:9090',
  reconnectMs: Number(process.env.DREAMWALKER_ROSBRIDGE_RECONNECT_MS || 2_000),
  frameId: process.env.DREAMWALKER_ROSBRIDGE_FRAME_ID || 'dreamwalker_map',
  linearThreshold: 0.25,
  angularThreshold: 0.25,
  quiet: false
};

const topicMap = {
  robotStateJson: '/dreamwalker/robot_state_json',
  robotPose2d: '/dreamwalker/robot_pose2d',
  robotPoseStamped: '/dreamwalker/robot_pose_stamped',
  robotWaypoint: '/dreamwalker/robot_waypoint',
  robotWaypointJson: '/dreamwalker/robot_waypoint_json',
  robotGoalPoseStamped: '/dreamwalker/robot_goal_pose_stamped',
  robotRouteJson: '/dreamwalker/robot_route_json',
  robotRoutePath: '/dreamwalker/robot_route_path',
  cameraCompressed: '/dreamwalker/camera/compressed',
  cameraInfo: '/dreamwalker/camera/camera_info',
  depthImage: '/dreamwalker/depth/image',
  cmdJson: '/dreamwalker/cmd_json',
  cmdPose2d: '/dreamwalker/cmd_pose2d',
  cmdWaypoint: '/dreamwalker/cmd_waypoint',
  cmdVel: '/dreamwalker/cmd_vel',
  requestState: '/dreamwalker/request_state'
};

const options = parseArgs(process.argv.slice(2));
let roboticsSocket = null;
let rosbridgeSocket = null;
let roboticsReconnectTimer = null;
let rosbridgeReconnectTimer = null;
let lastRobotState = null;
let shuttingDown = false;

connectRobotics();
connectRosbridge();

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

function parseArgs(argv) {
  const next = { ...defaults };

  while (argv.length) {
    const flag = argv.shift();

    if (flag === '--robotics') {
      next.roboticsUrl = normalizeRobotBridgeUrl(argv.shift() || defaults.roboticsUrl);
      continue;
    }

    if (flag === '--rosbridge') {
      const candidate = argv.shift() || defaults.rosbridgeUrl;
      next.rosbridgeUrl = String(candidate).replace(/\/$/, '');
      continue;
    }

    if (flag === '--reconnect-ms') {
      next.reconnectMs = Number(argv.shift() || defaults.reconnectMs);
      continue;
    }

    if (flag === '--frame-id') {
      next.frameId = argv.shift() || defaults.frameId;
      continue;
    }

    if (flag === '--quiet') {
      next.quiet = true;
      continue;
    }

    if (flag === '--help') {
      printHelp();
      process.exit(0);
    }

    throw new Error(`Unknown option: ${flag}`);
  }

  return next;
}

function printHelp() {
  console.log(`Dynamic Map Viewer ROSBridge Relay

Protocol: ${robotBridgeProtocolId}

Usage:
  npm run robotics:rosbridge -- [--robotics ws://127.0.0.1:8790/robotics] [--rosbridge ws://127.0.0.1:9090] [--frame-id dreamwalker_map]

Publishes to rosbridge:
  ${topicMap.robotStateJson}   std_msgs/String
  ${topicMap.robotPose2d}      geometry_msgs/Pose2D (Dynamic Map Viewer native plane)
  ${topicMap.robotPoseStamped} geometry_msgs/PoseStamped (ROS map frame)
  ${topicMap.robotWaypoint}    geometry_msgs/Point
  ${topicMap.robotWaypointJson} std_msgs/String
  ${topicMap.robotGoalPoseStamped} geometry_msgs/PoseStamped
  ${topicMap.robotRouteJson}   std_msgs/String
  ${topicMap.robotRoutePath}   nav_msgs/Path
  ${topicMap.cameraCompressed} sensor_msgs/CompressedImage
  ${topicMap.cameraInfo}       sensor_msgs/CameraInfo
  ${topicMap.depthImage}       sensor_msgs/Image

Subscribes from rosbridge:
  ${topicMap.cmdJson}          std_msgs/String(JSON v1 command)
  ${topicMap.cmdPose2d}        geometry_msgs/Pose2D
  ${topicMap.cmdWaypoint}      geometry_msgs/Point
  ${topicMap.cmdVel}           geometry_msgs/Twist
  ${topicMap.requestState}     std_msgs/Empty
`);
}

function log(message) {
  if (!options.quiet) {
    console.log(`[dreamwalker-rosbridge-relay] ${message}`);
  }
}

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isOpen(socket) {
  return socket?.readyState === WebSocket.OPEN;
}

function toMessageBuffer(payload) {
  if (Buffer.isBuffer(payload)) {
    return payload;
  }

  if (payload instanceof ArrayBuffer) {
    return Buffer.from(payload);
  }

  if (ArrayBuffer.isView(payload)) {
    return Buffer.from(payload.buffer, payload.byteOffset, payload.byteLength);
  }

  if (Array.isArray(payload)) {
    return Buffer.concat(payload.map((chunk) => toMessageBuffer(chunk)));
  }

  return Buffer.from(String(payload), 'utf8');
}

function toMessageText(payload) {
  return toMessageBuffer(payload).toString('utf8');
}

function normalizePositiveInteger(value, label) {
  const normalized = Number(value);

  if (!Number.isInteger(normalized) || normalized <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }

  return normalized;
}

function normalizePositiveNumber(value, label) {
  const normalized = Number(value);

  if (!Number.isFinite(normalized) || normalized <= 0) {
    throw new Error(`${label} must be a positive number`);
  }

  return normalized;
}

function normalizeFiniteVector(value, expectedLength, label) {
  if (
    !Array.isArray(value) ||
    value.length !== expectedLength ||
    value.some((entry) => !Number.isFinite(entry))
  ) {
    throw new Error(`${label} must be an array of ${expectedLength} finite numbers`);
  }

  return value.map((entry) => Number(entry));
}

function parseTimestampMs(value) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Number(value);
  }

  if (value instanceof Date) {
    const timestampMs = value.getTime();
    return Number.isFinite(timestampMs) ? timestampMs : null;
  }

  if (typeof value !== 'string' || !value.trim()) {
    return null;
  }

  const timestampMs = Date.parse(value);
  return Number.isFinite(timestampMs) ? timestampMs : null;
}

function scheduleReconnect(kind) {
  if (shuttingDown) {
    return;
  }

  const timerId = setTimeout(() => {
    if (kind === 'robotics') {
      roboticsReconnectTimer = null;
      connectRobotics();
      return;
    }

    rosbridgeReconnectTimer = null;
    connectRosbridge();
  }, options.reconnectMs);

  if (kind === 'robotics') {
    if (roboticsReconnectTimer) {
      clearTimeout(roboticsReconnectTimer);
    }
    roboticsReconnectTimer = timerId;
    return;
  }

  if (rosbridgeReconnectTimer) {
    clearTimeout(rosbridgeReconnectTimer);
  }
  rosbridgeReconnectTimer = timerId;
}

function clearReconnectTimers() {
  if (roboticsReconnectTimer) {
    clearTimeout(roboticsReconnectTimer);
    roboticsReconnectTimer = null;
  }

  if (rosbridgeReconnectTimer) {
    clearTimeout(rosbridgeReconnectTimer);
    rosbridgeReconnectTimer = null;
  }
}

function connectRobotics() {
  if (shuttingDown || isOpen(roboticsSocket)) {
    return;
  }

  log(`connecting robotics bridge ${options.roboticsUrl}`);
  const socket = new WebSocket(options.roboticsUrl);
  roboticsSocket = socket;

  socket.on('open', () => {
    log('robotics bridge connected');
    sendRobotCommand('request-state');
  });

  socket.on('message', (payload, isBinary) => {
    try {
      if (isBinary) {
        handleRoboticsBinaryMessage(toMessageBuffer(payload));
        return;
      }

      handleRoboticsMessage(toMessageText(payload));
    } catch (error) {
      log(`robotics message error: ${error instanceof Error ? error.message : String(error)}`);
    }
  });

  socket.on('error', (error) => {
    log(`robotics socket error: ${error instanceof Error ? error.message : String(error)}`);
  });

  socket.on('close', () => {
    if (roboticsSocket === socket) {
      roboticsSocket = null;
    }
    log('robotics bridge closed');
    scheduleReconnect('robotics');
  });
}

function connectRosbridge() {
  if (shuttingDown || isOpen(rosbridgeSocket)) {
    return;
  }

  log(`connecting rosbridge ${options.rosbridgeUrl}`);
  const socket = new WebSocket(options.rosbridgeUrl);
  rosbridgeSocket = socket;

  socket.on('open', () => {
    log('rosbridge connected');
    announceRosbridgeTopics();
    subscribeRosbridgeTopics();

    if (lastRobotState) {
      publishRobotState(lastRobotState);
      return;
    }

    sendRobotCommand('request-state');
  });

  socket.on('message', (payload, isBinary) => {
    try {
      if (isBinary) {
        log('ignoring unexpected binary frame from rosbridge');
        return;
      }

      handleRosbridgeMessage(toMessageText(payload));
    } catch (error) {
      log(`rosbridge message error: ${error instanceof Error ? error.message : String(error)}`);
    }
  });

  socket.on('error', (error) => {
    log(`rosbridge socket error: ${error instanceof Error ? error.message : String(error)}`);
  });

  socket.on('close', () => {
    if (rosbridgeSocket === socket) {
      rosbridgeSocket = null;
    }
    log('rosbridge closed');
    scheduleReconnect('rosbridge');
  });
}

function sendRobotCommand(type, fields = {}) {
  if (!isOpen(roboticsSocket)) {
    return false;
  }

  roboticsSocket.send(
    stringifyRobotBridgeMessage(type, fields, {
      source: robotBridgeRosbridgeSource
    })
  );
  return true;
}

function sendRobotCommandObject(message) {
  if (!isOpen(roboticsSocket)) {
    return false;
  }

  roboticsSocket.send(JSON.stringify(message));
  return true;
}

function sendRosbridge(frame) {
  if (!isOpen(rosbridgeSocket)) {
    return false;
  }

  rosbridgeSocket.send(JSON.stringify(frame));
  return true;
}

function advertiseRosbridgeTopic(topic, type) {
  return sendRosbridge({
    op: 'advertise',
    topic,
    type
  });
}

function subscribeRosbridgeTopic(topic) {
  return sendRosbridge({
    op: 'subscribe',
    topic
  });
}

function publishRosbridgeTopic(topic, msg) {
  return sendRosbridge({
    op: 'publish',
    topic,
    msg
  });
}

function announceRosbridgeTopics() {
  advertiseRosbridgeTopic(topicMap.robotStateJson, 'std_msgs/String');
  advertiseRosbridgeTopic(topicMap.robotPose2d, 'geometry_msgs/Pose2D');
  advertiseRosbridgeTopic(topicMap.robotPoseStamped, 'geometry_msgs/PoseStamped');
  advertiseRosbridgeTopic(topicMap.robotWaypoint, 'geometry_msgs/Point');
  advertiseRosbridgeTopic(topicMap.robotWaypointJson, 'std_msgs/String');
  advertiseRosbridgeTopic(topicMap.robotGoalPoseStamped, 'geometry_msgs/PoseStamped');
  advertiseRosbridgeTopic(topicMap.robotRouteJson, 'std_msgs/String');
  advertiseRosbridgeTopic(topicMap.robotRoutePath, 'nav_msgs/Path');
  advertiseRosbridgeTopic(topicMap.cameraCompressed, 'sensor_msgs/CompressedImage');
  advertiseRosbridgeTopic(topicMap.cameraInfo, 'sensor_msgs/CameraInfo');
  advertiseRosbridgeTopic(topicMap.depthImage, 'sensor_msgs/Image');
}

function subscribeRosbridgeTopics() {
  subscribeRosbridgeTopic(topicMap.cmdJson);
  subscribeRosbridgeTopic(topicMap.cmdPose2d);
  subscribeRosbridgeTopic(topicMap.cmdWaypoint);
  subscribeRosbridgeTopic(topicMap.cmdVel);
  subscribeRosbridgeTopic(topicMap.requestState);
}

function handleRoboticsMessage(rawMessage) {
  const { messageType, message } = parseRobotBridgeMessage(rawMessage);

  if (messageType === 'bridge-ready') {
    log(`robotics bridge ready from ${message.source ?? 'unknown'}`);
    sendRobotCommand('request-state');
    return;
  }

  if (messageType === 'bridge-error') {
    throw new Error(typeof message.error === 'string' ? message.error : 'bridge-error');
  }

  if (messageType !== 'robot-state') {
    return;
  }

  lastRobotState = message;
  publishRobotState(message);
}

function handleRoboticsBinaryMessage(payload) {
  const frame = parseBinaryFrameMessage(payload);

  if (frame.header.type === 'camera-frame') {
    publishCameraFrame(frame);
    return;
  }

  if (frame.header.type === 'depth-frame') {
    publishDepthFrame(frame);
    return;
  }

  throw new Error(`unsupported robotics binary frame type: ${frame.header.type}`);
}

function publishRobotState(message) {
  publishRosbridgeTopic(topicMap.robotStateJson, {
    data: JSON.stringify(message)
  });

  const pose2d = toPose2D(message);
  if (pose2d) {
    publishRosbridgeTopic(topicMap.robotPose2d, pose2d);
  }

  const poseStamped = toPoseStamped(message);
  if (poseStamped) {
    publishRosbridgeTopic(topicMap.robotPoseStamped, poseStamped);
  }

  publishRosbridgeTopic(topicMap.robotWaypointJson, {
    data: JSON.stringify(message.waypoint ?? null)
  });

  if (message.waypoint?.position) {
    publishRosbridgeTopic(topicMap.robotWaypoint, toPoint(message.waypoint.position));

    const goalPoseStamped = toGoalPoseStamped(message);
    if (goalPoseStamped) {
      publishRosbridgeTopic(topicMap.robotGoalPoseStamped, goalPoseStamped);
    }
  }

  publishRosbridgeTopic(topicMap.robotRouteJson, {
    data: JSON.stringify(message.route ?? [])
  });

  const routePath = toRoutePath(message);
  if (routePath) {
    publishRosbridgeTopic(topicMap.robotRoutePath, routePath);
  }
}

function publishCameraFrame(frame) {
  publishRosbridgeTopic(topicMap.cameraCompressed, toCompressedImage(frame));
  publishRosbridgeTopic(topicMap.cameraInfo, toCameraInfo(frame));
}

function publishDepthFrame(frame) {
  publishRosbridgeTopic(topicMap.depthImage, toDepthImage(frame));
}

function handleRosbridgeMessage(rawMessage) {
  const parsed = JSON.parse(String(rawMessage));

  if (!isRecord(parsed) || parsed.op !== 'publish' || typeof parsed.topic !== 'string') {
    return;
  }

  if (parsed.topic === topicMap.requestState) {
    sendRobotCommand('request-state');
    return;
  }

  if (parsed.topic === topicMap.cmdJson) {
    handleRosbridgeJsonCommand(parsed.msg);
    return;
  }

  if (parsed.topic === topicMap.cmdPose2d) {
    const pose2d = normalizePose2D(parsed.msg);
    if (!pose2d) {
      throw new Error('cmd_pose2d payload is invalid');
    }

    sendRobotCommand('set-pose', {
      pose: {
        position: [pose2d.x, 0, pose2d.y],
        yawDegrees: radiansToDegrees(pose2d.theta)
      },
      resetRoute: pose2d.resetRoute !== false,
      clearWaypoint: Boolean(pose2d.clearWaypoint)
    });
    return;
  }

  if (parsed.topic === topicMap.cmdWaypoint) {
    const point = normalizePoint(parsed.msg);
    if (!point) {
      throw new Error('cmd_waypoint payload is invalid');
    }

    sendRobotCommand('set-waypoint', {
      position: [point.x, point.y, point.z]
    });
    return;
  }

  if (parsed.topic === topicMap.cmdVel) {
    const action = mapTwistToAction(parsed.msg);
    if (!action) {
      return;
    }

    sendRobotCommand('teleop', {
      action
    });
  }
}

function handleRosbridgeJsonCommand(rawMessage) {
  const payload = isRecord(rawMessage) && 'data' in rawMessage ? rawMessage.data : rawMessage;

  if (typeof payload === 'string') {
    const trimmed = payload.trim();

    if (!trimmed) {
      return;
    }

    if (['forward', 'backward', 'turn-left', 'turn-right'].includes(trimmed)) {
      sendRobotCommand('teleop', {
        action: trimmed
      });
      return;
    }
  }

  const normalizedMessage = normalizeRobotBridgeMessage(payload, {
    fallbackSource: robotBridgeRosbridgeSource
  });
  sendRobotCommandObject(normalizedMessage);
}

function parseBinaryFrameEnvelope(payload, label) {
  if (!Buffer.isBuffer(payload) || payload.length < 5) {
    throw new Error(`${label} payload is too short`);
  }

  const headerLength = payload.readUInt32LE(0);
  const headerEnd = 4 + headerLength;

  if (headerLength <= 0 || headerEnd > payload.length) {
    throw new Error(`${label} header length is invalid`);
  }

  return {
    header: JSON.parse(payload.subarray(4, headerEnd).toString('utf8')),
    payload: payload.subarray(headerEnd)
  };
}

function parseBinaryFrameMessage(payload) {
  const envelope = parseBinaryFrameEnvelope(payload, 'robotics binary frame');

  if (envelope.header?.type === 'camera-frame') {
    return parseCameraFrameMessage(envelope);
  }

  if (envelope.header?.type === 'depth-frame') {
    return parseDepthFrameMessage(envelope);
  }

  throw new Error(
    `unsupported robotics binary frame header type: ${String(envelope.header?.type ?? 'unknown')}`
  );
}

function parseCameraFrameMessage(envelope) {
  if (!envelope.payload.length) {
    throw new Error('camera-frame JPEG payload is empty');
  }

  return {
    header: normalizeCameraFrameHeader(envelope.header),
    jpegPayload: envelope.payload
  };
}

function parseDepthFrameMessage(envelope) {
  const header = normalizeDepthFrameHeader(envelope.header);
  const expectedPayloadLength = header.width * header.height * 4;

  if (envelope.payload.length !== expectedPayloadLength) {
    throw new Error(
      `depth-frame payload length mismatch: expected ${expectedPayloadLength}, got ${envelope.payload.length}`
    );
  }

  return {
    header,
    depthPayload: envelope.payload
  };
}

function normalizeBinaryFrameHeader(value, label, expectedType) {
  if (!isRecord(value) || value.type !== expectedType) {
    throw new Error(`${label} header type is invalid`);
  }

  const timestampMs = parseTimestampMs(value.timestamp);
  if (!Number.isFinite(timestampMs)) {
    throw new Error(`${label} timestamp is invalid`);
  }

  const cameraId = typeof value.cameraId === 'string' ? value.cameraId.trim() : '';
  if (!cameraId) {
    throw new Error(`${label} cameraId is required`);
  }

  return {
    type: expectedType,
    timestamp: value.timestamp,
    timestampMs,
    cameraId,
    width: normalizePositiveInteger(value.width, `${label} width`),
    height: normalizePositiveInteger(value.height, `${label} height`),
    fov: normalizePositiveNumber(value.fov, `${label} fov`),
    pose: normalizeCameraFramePose(value.pose)
  };
}

function normalizeCameraFrameHeader(value) {
  return normalizeBinaryFrameHeader(value, 'camera-frame', 'camera-frame');
}

function normalizeDepthFrameHeader(value) {
  const header = normalizeBinaryFrameHeader(value, 'depth-frame', 'depth-frame');
  const encoding = typeof value.encoding === 'string' ? value.encoding.trim() : '';

  if (encoding !== '32FC1') {
    throw new Error('depth-frame encoding must be 32FC1');
  }

  return {
    ...header,
    encoding,
    nearClip: normalizePositiveNumber(value.nearClip, 'depth-frame nearClip'),
    farClip: normalizePositiveNumber(value.farClip, 'depth-frame farClip')
  };
}

function normalizeCameraFramePose(value) {
  if (!isRecord(value)) {
    throw new Error('camera-frame pose is required');
  }

  const position = normalizeFiniteVector(value.position, 3, 'camera-frame pose.position');
  const orientation = normalizeFiniteVector(value.orientation, 4, 'camera-frame pose.orientation');

  return {
    position,
    orientation,
    rosPosition: toRosMapPosition(position)
  };
}

function toCompressedImage(frame) {
  return {
    header: buildHeader(frame.header.timestampMs),
    format: 'jpeg',
    data: frame.jpegPayload.toString('base64')
  };
}

function toCameraInfo(frame) {
  const { width, height, fov, timestampMs } = frame.header;
  const fovRadians = degreesToRadians(fov);
  const fy = height / (2 * Math.tan(fovRadians / 2));
  const fx = fy * (width / height);
  const cx = width / 2;
  const cy = height / 2;

  return {
    header: buildHeader(timestampMs),
    width,
    height,
    distortion_model: 'plumb_bob',
    d: [0, 0, 0, 0, 0],
    k: [fx, 0, cx, 0, fy, cy, 0, 0, 1],
    r: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    p: [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0],
    binning_x: 0,
    binning_y: 0,
    roi: {
      x_offset: 0,
      y_offset: 0,
      height: 0,
      width: 0,
      do_rectify: false
    }
  };
}

function toDepthImage(frame) {
  return {
    header: buildHeader(frame.header.timestampMs),
    height: frame.header.height,
    width: frame.header.width,
    encoding: frame.header.encoding,
    is_bigendian: 0,
    step: frame.header.width * 4,
    data: frame.depthPayload.toString('base64')
  };
}

function toPose2D(message) {
  const position = message?.pose?.position;
  const yawDegrees = Number(message?.pose?.yawDegrees);

  if (!Array.isArray(position) || position.length < 3 || !Number.isFinite(yawDegrees)) {
    return null;
  }

  return {
    x: Number(position[0]),
    y: Number(position[2]),
    theta: degreesToRadians(yawDegrees)
  };
}

function toPoseStamped(message) {
  const position = message?.pose?.position;
  const yawDegrees = Number(message?.pose?.yawDegrees);

  if (!Array.isArray(position) || position.length < 3 || !Number.isFinite(yawDegrees)) {
    return null;
  }

  return buildPoseStamped(position, yawDegrees);
}

function toGoalPoseStamped(message) {
  const currentPosition = message?.pose?.position;
  const fallbackYawDegrees = Number(message?.pose?.yawDegrees);
  const waypointPosition = message?.waypoint?.position;

  if (
    !Array.isArray(currentPosition) ||
    currentPosition.length < 3 ||
    !Array.isArray(waypointPosition) ||
    waypointPosition.length < 3 ||
    !Number.isFinite(fallbackYawDegrees)
  ) {
    return null;
  }

  const goalYawDegrees = deriveWorldYawDegreesFromPositions(
    currentPosition,
    waypointPosition,
    fallbackYawDegrees
  );
  return buildPoseStamped(waypointPosition, goalYawDegrees);
}

function toRoutePath(message) {
  const route = Array.isArray(message?.route) ? message.route : [];
  const fallbackYawDegrees = Number(message?.pose?.yawDegrees ?? 0);

  if (!route.length) {
    return null;
  }

  return {
    header: buildHeader(),
    poses: route.map((position, index) => {
      const previous = route[index - 1] ?? null;
      const next = route[index + 1] ?? null;
      const headingTarget = next ?? previous ?? position;
      const nodeYawDegrees = deriveWorldYawDegreesFromPositions(
        position,
        headingTarget,
        fallbackYawDegrees
      );

      return buildPoseStamped(position, nodeYawDegrees);
    })
  };
}

function toPoint(position) {
  return {
    x: Number(position[0]),
    y: Number(position[1] ?? 0),
    z: Number(position[2] ?? 0)
  };
}

function normalizePose2D(value) {
  if (!isRecord(value)) {
    return null;
  }

  const x = Number(value.x);
  const y = Number(value.y);
  const theta = Number(value.theta ?? 0);

  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(theta)) {
    return null;
  }

  return {
    ...value,
    x,
    y,
    theta
  };
}

function normalizePoint(value) {
  if (!isRecord(value)) {
    return null;
  }

  const x = Number(value.x);
  const y = Number(value.y ?? 0);
  const z = Number(value.z);

  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    return null;
  }

  return { x, y, z };
}

function mapTwistToAction(value) {
  if (!isRecord(value) || !isRecord(value.linear) || !isRecord(value.angular)) {
    return null;
  }

  const linearX = Number(value.linear.x ?? 0);
  const angularZ = Number(value.angular.z ?? 0);

  if (!Number.isFinite(linearX) || !Number.isFinite(angularZ)) {
    return null;
  }

  const linearMagnitude = Math.abs(linearX);
  const angularMagnitude = Math.abs(angularZ);

  if (linearMagnitude < options.linearThreshold && angularMagnitude < options.angularThreshold) {
    return null;
  }

  if (linearMagnitude >= angularMagnitude) {
    return linearX >= 0 ? 'forward' : 'backward';
  }

  return angularZ >= 0 ? 'turn-left' : 'turn-right';
}

function degreesToRadians(value) {
  return (Number(value) * Math.PI) / 180;
}

function radiansToDegrees(value) {
  return (Number(value) * 180) / Math.PI;
}

function buildHeader(timestamp = Date.now(), frameId = options.frameId) {
  const timestampMs = parseTimestampMs(timestamp) ?? Date.now();
  return {
    frame_id: typeof frameId === 'string' && frameId.trim() ? frameId.trim() : options.frameId,
    stamp: {
      sec: Math.floor(timestampMs / 1_000),
      nanosec: (timestampMs % 1_000) * 1_000_000
    }
  };
}

function buildPoseStamped(position, yawDegrees) {
  return {
    header: buildHeader(),
    pose: {
      position: toRosMapPosition(position),
      orientation: quaternionFromYaw(worldYawDegreesToRosYawRadians(yawDegrees))
    }
  };
}

function toRosMapPosition(position) {
  return {
    x: Number(position[0]),
    y: Number(position[2]),
    z: Number(position[1] ?? 0)
  };
}

function worldYawDegreesToRosYawRadians(yawDegrees) {
  const yawRadians = degreesToRadians(yawDegrees);
  return Math.atan2(-Math.cos(yawRadians), -Math.sin(yawRadians));
}

function quaternionFromYaw(yawRadians) {
  const halfYaw = yawRadians / 2;
  return {
    x: 0,
    y: 0,
    z: Math.sin(halfYaw),
    w: Math.cos(halfYaw)
  };
}

function deriveWorldYawDegreesFromPositions(startPosition, targetPosition, fallbackYawDegrees) {
  if (!Array.isArray(startPosition) || !Array.isArray(targetPosition)) {
    return fallbackYawDegrees;
  }

  const dx = Number(targetPosition[0]) - Number(startPosition[0]);
  const dz = Number(targetPosition[2]) - Number(startPosition[2]);

  if (!Number.isFinite(dx) || !Number.isFinite(dz) || Math.hypot(dx, dz) < 0.0001) {
    return fallbackYawDegrees;
  }

  return radiansToDegrees(Math.atan2(-dx, -dz));
}

function shutdown() {
  shuttingDown = true;
  clearReconnectTimers();
  roboticsSocket?.close();
  rosbridgeSocket?.close();
  setTimeout(() => process.exit(0), 10);
}
