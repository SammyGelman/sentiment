const express = require('express');
const bodyParser = require('body-parser');
const {spawn} = require('child_process')
const port = 3000

const app = express();

let userGoal = 'Sarcasm';
let dataToSend;

app.use(
  bodyParser.urlencoded({
    extended: false,
  })
);

app.use(express.static('public'));

app.get('/', (req, res) => {
  res.send(`
    <html>
      <head>
        <link rel="stylesheet" href="styles.css">
      </head>
      <body>
        <section>
          <h2>Sentiment Detection</h2>
          <h3>${userGoal}</h3>
        </section>
        <form action="/set-sentence" method="GET">
        <form action="/store-goal" method="POST">
          <div class="form-control">
            <label>Type Sentance Here</label>
            <input type="text" name="goal">
          </div>
          <button>Check for sentiment</button>
        </form>
      </body>
    </html>
  `);
});

// app.post('/store-goal', (req, res) => {
//   const enteredGoal = req.body.goal;
//   res.redirect('/');
// });



app.get('/set-sentence', (req, res) => {
  const findSentence = (Sentence) => {
        return 'python_test.py "' + Sentence + ' proof it worked"'  
  };
  dataToSend = req.params
  // const sentence = findSentence(req.body.goal);
  // // userGoal = findSentence(enteredGoal) 
  // const python = spawn('python', [sentence]);  
  // python.stdout.on('data', function(data) {
  //     dataToSend = req.params.data.toString();
  // });  
  // python.on('close');
  res.send(dataToSend)
});

app.post('/store-goal', (req, res) => { 
  dataToSend = req.body.goal;
  userGoal = dataToSend;
  console.log();
  res.redirect('/');
});
//app.listen(80);
app.listen(port);
