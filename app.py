from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from load_and_run_embedding import sentiment

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)


class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    # sarcasm = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Sentence %r>' % self.id


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        sentence_content = request.form['content']
        sarcasm = str(sentiment(sentence_content))

        new_sentence = Todo(content=sarcasm)
        try:
            db.session.add(new_sentence)
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue checking that sentence'

        # sarcasm_content = request.form['sarcasm']
        # new_sarcasm = sentiment(new_sentence)
        #
        # try:
        #     db.session.add(new_sarcasm)
        #     db.session.commit()
        #     return redirect('/')
        # except:
        #     return "There was a problem measuring the sarcasm"
    else:
        tasks = Todo.query.order_by(Todo.date_created).all()
        return render_template('index.html', tasks=tasks)


@app.route('/delete/<int:id>')
def delete(id):
    sentence_to_delete = Todo.query.get_or_404(id)

    try:
        db.session.delete(sentence_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting that task'


# @app.route('/update/<int:id>', methods=['GET', 'POST'])
# def update(id):
#     task = Todo.query.get_or_404(id)
#
#     if request.method == 'POST':
#         task.content = request.form['content']
#
#         try:
#             db.session.commit()
#             return redirect('/')
#         except:
#             return 'There was an issue updating your task'
#
#     else:
#         return render_template('update.html', task=task)
#

if __name__ == "__main__":
    app.run(debug=True)
