// Setup basic express server
var express = require('express');
var app = express();
var server = require('http').createServer(app);
var io = require('socket.io')(server);
var redis = require('socket.io-redis');
var port = process.env.PORT;
var serverName = process.env.NAME || 'Unknown';

io.adapter(redis({ host: 'redis', port: 6379 }));

server.listen(port, function () {
  console.log('Server listening at port %d', port);
});

// Routing
app.use(express.static(__dirname + '/public'));

// Health check
app.head('/health', function (req, res) {
  res.sendStatus(200);
});

