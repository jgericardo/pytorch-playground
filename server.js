/** PyTorch Playground
 * @version v1.0.0
 * @license Apache-2.0
 * @copyright Justin Gerard E. Ricardo
 * @link https://github.com/jgericardo/wishnet-playground

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================*/
// Server creation and server message handler. 

// Node.JS module imports
const express = require('express');
const path = require('path');
const open = require('open');
const WebSocket = require("ws");
const http = require("http");
const {spawn, ChildProcess} = require('child_process');

// Express variables
const app = express();

// Server and websocket variables
const server = http.createServer(app);
const wss = new WebSocket.Server({server});
const port = 8080;
const scriptPath = "scripts/train_ann.py";


/** Helper function to run script as a child process thread.
 * @param dataset {String}
 * @param hiddenNodes {String}
 * @param learningRate {String}
 * @param epochs {String}
 * @return {ChildProcess}
 */
function runScript(
  dataset, hiddenNodes, learningRate, epochs
) {
  absScriptPath = path.join(__dirname, scriptPath);

  return spawn(
    'python',
    [
      "-u", absScriptPath,
      "--dataset", dataset,
      "--learning_rate", learningRate,
      "--hidden_nodes", hiddenNodes,
      "--epochs", epochs,
      "--script_dir", __dirname
    ]
  );
}

/** Running of script within the server websocket
 * @param id {String}
 * @param ws {WebSocket}
 * @param dataset {String}
 * @param hiddenNodes {String}
 * @param learningRate {String}
 * @param epochs {String}
 */
function runScriptInWebSocket(
  id, ws, dataset, hiddenNodes, learningRate, epochs
){
  // create child process
  const child = runScript(
    dataset, hiddenNodes, learningRate, epochs
  );

  // listen for Python messages
  child.stdout.on('data', (data) => {
    ws.send(`${data}`);
  });
  child.stderr.on('data', (data) => {
    ws.send(`${id}:error:\n${data}`);
  });
  child.on('close', () => {
    ws.send(`${id}:done`);
  });
}

// router requests the root page
// thus, index.html page would be rendered.
app.get('/', function(req, res){
  console.log(__dirname);
  res.sendFile(path.join(__dirname, '/dist/index.html'));
});

// serve the static files needed by index.html
// such as javascript, css, and image files.
app.use(express.static(path.join(__dirname, '/dist')));


// define handling behavior of server events:
// - on server connection initialziation
// - on message receive from client
let id = 1
wss.on('connection', (ws) => {
  let thisId = id++;
  ws.send(`[SERVER][INFO] Current connection id -> ${thisId}`)

  ws.on('message', (message) => {
    ws.send(`[SERVER][INFO] Client sent -> ${message}`);
    message = JSON.parse(message.replace(/'/g, '"'));
    
    if (message["command"] === "run") {
      dataset = message["dataset"];
      hiddenNodes = +message["hiddenNodes"];
      learningRate = +message["learningRate"];
      epochs = +message["epochs"];

      runScriptInWebSocket(
        thisId, ws, dataset, hiddenNodes, learningRate, epochs
      );
    }
    else if (message["command"] === "close") {
      ws.close();
    }
  });

  ws.send('[SERVER][INFO] Connection with WebSocket server initialized.');
});

// start server
server.listen(port, () => {
  console.log('+--------------------------')
  console.log(' PID %d', process.pid)
  console.log(' Listening on port', port)
  console.log('+--------------------------')
});

// open the client in the default browser's tab
open("http://localhost:8080");