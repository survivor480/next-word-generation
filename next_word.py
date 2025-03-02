import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pandas import SparseDtype





file = open('archive/processed_data.csv')
stories = []
num_of_stories = 10
for story in file:
    stories.append(story)
    num_of_stories -= 1
    if num_of_stories == 0:
        break


# initialize a tokenizer to convert words into integer tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(stories)

# get the word-to_index dictionary
word_index = tokenizer.word_index

print(len(word_index))



# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(word_index)+1, output_dim=50),
    tf.keras.layers.SimpleRNN(units=100, return_sequences=False),
    tf.keras.layers.Dense(len(word_index)+1, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# initializing the model
model.build(input_shape=(None,5))

# Print the summary of the model
model.summary()



for story in stories:
  # convert the stories to sequences of word indices
  sequences = tokenizer.texts_to_sequences([story])

  # create input output pairs 5 words as input next word as output
  x=[]
  y=[]
  for seq in sequences:
    for i in range(5,len(seq)):
      x.append(seq[i-5:i])
      y.append(seq[i])
  # convert to numpy arrays
  x = np.array(x)
  y = np.array(y)

  # pad sequences if necessary
  x = pad_sequences(x, maxlen=5, padding='pre')

  # one hot encode the target labels(y)
  y = tf.keras.utils.to_categorical(y, num_classes=len(word_index) + 1)

  # training the model
  model.fit(x,y,epochs=20, batch_size=32)

  del x
  del y



  # Assuming 'model' and 'tokenizer' are defined from the previous code.
# Also assuming 'word_index' is available.

def predict_next_word(seed_text):
    """Predicts the next word based on the input seed text."""

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=5, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)

    for word, index in word_index.items():
        if index == predicted:
            return word
    return None  # Return None if no matching word is found


# Example usage
seed_text = "plumb by an old blast"
predicted_word = predict_next_word(seed_text)
print(f"Seed text: {seed_text}")
print(f"Predicted next word: {predicted_word}")




#prdicting the next 50 words
seed_text = "plumb by an old blast"
text = "plumb by an old blast"
next_word = ''
for i in range(50):
  next_word = predict_next_word(seed_text)
  seed_text = ' '.join(seed_text.split(' ')[1:])
  seed_text += ' ' + next_word
  text += ' ' + next_word

print(text)


model.save('next_word.keras')