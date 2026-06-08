import { WebSocket } from 'ws';
import {
  normalizeRobotBridgeUrl,
  parseRobotBridgeMessage,
  robotBridgeCliSource,
  robotBridgeDefaultUrl,
  robotBridgeProtocolId,
  stringifyRobotBridgeMessage
} from '../src/robotics-bridge.js';

const args = process.argv.slice(2);
const options = {
  url: robotBridgeDefaultUrl,
  timeoutMs: 10_000
};

while (args.length && args[0].startsWith('--')) {
  const flag = args.shift();

  if (flag === '--url') {
    options.url = normalizeRobotBridgeUrl(args.shift() || robotBridgeDefaultUrl);
    continue;
  }

  if (flag === '--timeout') {
    options.timeoutMs = Number(args.shift() || 10_000);
    continue;
  }

  if (flag === '--help') {
    printHelp();
    process.exit(0);
  }

  throw new Error(`Unknown option: ${flag}`);
}

const command = args.shift() || 'help';

if (command === 'help') {
  printHelp();
  process.exit(0);
}

main().catch((error) => {
  console.error(`[dreamwalker-robotics-cli] ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
});

async function main() {
  if (command === 'watch') {
    await runWatch();
    return;
  }

  const socket = await openSocket();

  try {
    switch (command) {
      case 'request-state':
        socket.send(stringifyRobotBridgeMessage('request-state', {}, { source: robotBridgeCliSource }));
        printJson(
          await waitForMessage(socket, (message) => message.type === 'robot-state', options.timeoutMs)
        );
        break;

      case 'teleop':
        socket.send(
          stringifyRobotBridgeMessage(
            'teleop',
            { action: requireEnumArg(args.shift(), ['forward', 'backward', 'turn-left', 'turn-right']) },
            { source: robotBridgeCliSource }
          )
        );
        break;

      case 'set-pose':
        socket.send(
          stringifyRobotBridgeMessage(
            'set-pose',
            {
              pose: {
                position: [
                  requireNumberFlag(args, '--x'),
                  readNumberFlag(args, '--y', 0),
                  requireNumberFlag(args, '--z')
                ],
                yawDegrees: readNumberFlag(args, '--yaw', 0)
              },
              resetRoute: readBooleanFlag(args, '--reset-route'),
              clearWaypoint: readBooleanFlag(args, '--clear-waypoint')
            },
            { source: robotBridgeCliSource }
          )
        );
        break;

      case 'set-waypoint':
        socket.send(
          stringifyRobotBridgeMessage(
            'set-waypoint',
            {
              position: [
                requireNumberFlag(args, '--x'),
                readNumberFlag(args, '--y', 0),
                requireNumberFlag(args, '--z')
              ]
            },
            { source: robotBridgeCliSource }
          )
        );
        break;

      case 'clear-waypoint':
      case 'clear-route':
      case 'reset-pose':
        socket.send(stringifyRobotBridgeMessage(command, {}, { source: robotBridgeCliSource }));
        break;

      case 'set-camera':
        socket.send(
          stringifyRobotBridgeMessage(
            'set-camera',
            { cameraId: requireStringArg(args.shift(), 'camera id') },
            { source: robotBridgeCliSource }
          )
        );
        break;

      default:
        throw new Error(`Unknown command: ${command}`);
    }
  } finally {
    socket.close();
  }
}

function printHelp() {
  console.log(`Dynamic Map Viewer Robotics CLI

Protocol: ${robotBridgeProtocolId}
Default URL: ${robotBridgeDefaultUrl}

Usage:
  npm run robotics:client -- [--url ws://127.0.0.1:8790/robotics] <command>

Commands:
  request-state
  watch
  teleop <forward|backward|turn-left|turn-right>
  set-pose --x <n> --z <n> [--y <n>] [--yaw <deg>] [--reset-route] [--clear-waypoint]
  set-waypoint --x <n> --z <n> [--y <n>]
  clear-waypoint
  clear-route
  reset-pose
  set-camera <front|chase|top>
`);
}

function printJson(value) {
  console.log(JSON.stringify(value, null, 2));
}

function requireStringArg(value, label) {
  if (typeof value === 'string' && value.trim()) {
    return value.trim();
  }

  throw new Error(`${label} is required`);
}

function requireEnumArg(value, candidates) {
  const normalized = requireStringArg(value, 'enum argument');

  if (candidates.includes(normalized)) {
    return normalized;
  }

  throw new Error(`Expected one of: ${candidates.join(', ')}`);
}

function readFlagIndex(queue, flag) {
  return queue.findIndex((value) => value === flag);
}

function readNumberFlag(queue, flag, defaultValue) {
  const flagIndex = readFlagIndex(queue, flag);

  if (flagIndex === -1) {
    return defaultValue;
  }

  const [removedFlag] = queue.splice(flagIndex, 1);
  const nextValue = queue.splice(flagIndex, 1)[0];
  const parsed = Number(nextValue);

  if (!Number.isFinite(parsed)) {
    throw new Error(`${removedFlag} requires a finite number`);
  }

  return parsed;
}

function requireNumberFlag(queue, flag) {
  const value = readNumberFlag(queue, flag, Number.NaN);

  if (!Number.isFinite(value)) {
    throw new Error(`${flag} is required`);
  }

  return value;
}

function readBooleanFlag(queue, flag) {
  const flagIndex = readFlagIndex(queue, flag);

  if (flagIndex === -1) {
    return false;
  }

  queue.splice(flagIndex, 1);
  return true;
}

function openSocket() {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(options.url);
    const timeoutId = setTimeout(() => {
      socket.terminate();
      reject(new Error(`robotics bridge open timed out: ${options.url}`));
    }, options.timeoutMs);

    socket.once('open', () => {
      clearTimeout(timeoutId);
      resolve(socket);
    });

    socket.once('error', (error) => {
      clearTimeout(timeoutId);
      reject(error);
    });
  });
}

function waitForMessage(socket, predicate, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      cleanup();
      reject(new Error('robotics bridge response timed out'));
    }, timeoutMs);

    function cleanup() {
      clearTimeout(timeoutId);
      socket.off('message', handleMessage);
      socket.off('error', handleError);
    }

    function handleError(error) {
      cleanup();
      reject(error);
    }

    function handleMessage(buffer) {
      try {
        const { message } = parseRobotBridgeMessage(buffer.toString());
        if (!predicate(message)) {
          return;
        }

        cleanup();
        resolve(message);
      } catch (error) {
        cleanup();
        reject(error);
      }
    }

    socket.on('message', handleMessage);
    socket.on('error', handleError);
  });
}

async function runWatch() {
  const socket = await openSocket();

  socket.on('message', (buffer) => {
    try {
      const { message } = parseRobotBridgeMessage(buffer.toString());
      printJson(message);
    } catch (error) {
      console.error(
        `[dreamwalker-robotics-cli] ${error instanceof Error ? error.message : String(error)}`
      );
    }
  });

  socket.on('close', () => {
    process.exit(0);
  });

  socket.on('error', (error) => {
    console.error(`[dreamwalker-robotics-cli] ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  });

  socket.send(stringifyRobotBridgeMessage('request-state', {}, { source: robotBridgeCliSource }));
  console.error(`[dreamwalker-robotics-cli] watching ${options.url}`);
}
