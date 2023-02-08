const express = require('express');
const bodyParser = require('body-parser');
const {spawn} = require('child_process')
const port = 3000

const app = express();

let userGoal = 'Sarcasm';

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
        <form action="/store-goal" method="POST">
          <div class="form-control">
            <label>Type Sentence Here</label>
            <input type="text" name="goal">
          </div>
          <button>Check for sentiment</button>
        </form>
      </body>
    </html>
  `);
});

app.post('/store-goal', (req, res) => {
  function findSentence (input) {
        return 'load_and_run_embedding.py "' + input + '"'   
  };
  const sentence = findSentence(req.body.goal);
  console.log(sentence)
  
  const python = spawn('docker', ['exec','-it','alias','python',sentence]);  
  python.stdout.on('data', (data) => {
  console.log(data.toString());
  userGoal = data.toString();
  });  
  python.on('close', (code) => {
  res.redirect('/')
  });
});

// app.post('/store-goal', (req, res) => {
//   function findSentence (input) {
//         return 'load_and_run_embedding.py "' + input + '"'   
//   };
//   const sentence = findSentence(req.body.goal);
//   console.log(sentence)
//   
//   const python = spawn('/usr/bin/python3', [sentence], {shell: true});  
//   python.stdout.on('data', (data) => {
//   console.log(data.toString());
//   userGoal = data.toString();
//   });  
//   python.on('close', (code) => {
//   res.redirect('/')
//   });
// });

app.listen(80);
// app.listen(port);
