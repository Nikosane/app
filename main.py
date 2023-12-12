import torch
import nltk
import json
import numpy as np
import random
import torch.nn as nn
from kivymd.app import MDApp 
from kivy.lang import Builder 
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager ,Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.label import MDLabel
from kivy.properties import StringProperty , NumericProperty
from kivy.metrics import dp
from kivy.clock import Clock
from nltk.stem.porter import PorterStemmer
#--------------------------------------------------------------------
#tokenization
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[index] = 1.0

    return bag

#----------------------------------------------------------------------------
# neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(input_size, hidden_size)
        self.l3 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        return out

#--------------------------------------------------------------------------
# chatbot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_response(msg):
    sentence = tokenize(msg)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()].strip()

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]


    

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                responses = intent['responses']
                non_empty_resposnses = [response for response in responses if response.strip()]
                if non_empty_resposnses:
                    response = random.choice(non_empty_resposnses)
                    print(f"Bot Response: {response}")
                    return response , prob.item(), tag

    response = "I'm not sure what you're asking . Can you please rephrase?"
    print(f"Bot Response: {response}")
    return response , prob.item(), tag if tag else None 



#------------------------------------------------------------------------------



Window.size = (350,550)

class MainScreen(Screen):
    pass

class ChatScreen(Screen):
    pass

class Command(MDLabel):
    text = StringProperty()
    size_hint_x = NumericProperty()
    halign = StringProperty
    font_size = 17

class Response(MDLabel):
    text = StringProperty()
    size_hint_x = NumericProperty()
    halign = StringProperty
    font_size = 17


class ChatBot(MDApp):

    def change_screen(self,name):
        screen_manager.current = name 

    def build(self):

        global screen_manager
        screen_manager = ScreenManager()
        screen_manager.add_widget(Builder.load_file("Main.kv"))
        screen_manager.add_widget(Builder.load_file("chat.kv"))
        # self.theme_cls.theme_style = "Dark"
        return screen_manager 

    def bot_name(self):
        if screen_manager.get_screen('main').bot_name.text != "":
            screen_manager.get_screen('chat').bot_name.text = screen_manager.get_screen('main').bot_name.text
            screen_manager.current = "chat"

    def _on_enter_pressed(self, event):
        msg = screen_manager.get_screen('chat')

    def response(self, *args):
        user_input = screen_manager.get_screen('chat').text_input.text
        if user_input:
            response, _, _ = get_response(user_input)
            print(f"Bot Response: {response}")
            chat_list = screen_manager.get_screen('chat').chat_list
            chat_list.add_widget(Response(text=response, size_hint_x=.75))
            screen_manager.get_screen('chat').text_input.text = ""
    

    def send(self):
        global size, halign , value
        if screen_manager.get_screen('chat').text_input != "":
            value = screen_manager.get_screen('chat').text_input.text
            if len(value)<6:
                size = .22
                halign = "center"
            elif len(value) <11:
                size = .32
                halign = "center"
            elif len(value) <16:
                size = .45
                halign = "center"
            elif len(value) <21:
                size = .58
                halign = "center"
            elif len(value) <26:
                size = .71
                halign = "center"
            else:
                size = .77
                halign = "left "
            screen_manager.get_screen('chat').chat_list.add_widget(Command(text=value, size_hint_x=size , halign=halign))
            Clock.schedule_once(self.response, 2)
            screen_manager.get_screen('chat').text_input.text=""
        # return size , halign , value


    def send_message(self, text_input):
        user_message = text_input.text
        response = ""
        
        if user_message:
            global size, halign, value
            if screen_manager.get_screen('chat').text_input != "":
                value = screen_manager.get_screen('chat').text_input.text
                if len(value) < 6:
                    size = .16
                    halign = "center"
                elif len(value) < 11:
                    size = .22
                    halign = "center"
                elif len(value) < 16:
                    size = .35
                    halign = "center"
                elif len(value) < 21:
                    size = .48
                    halign = "center"
                elif len(value) < 26:
                    size = .51
                    halign = "center"
                else:
                    size = .67
                    halign = "left"
                
            chat_screen = screen_manager.get_screen('chat')
            chat_list = chat_screen.chat_list
            command_widget = Command(text=user_message, size_hint_x=size, halign="right")
            chat_list.add_widget(command_widget)

    
            response, _, _ = get_response(user_message)
            

                
            # Apply similar size conditions for the Response text
            if len(response) < 6:
                response_size = .16
                response_halign = "center"
            elif len(response) < 11:
                response_size = .22
                response_halign = "center"
            elif len(response) < 16:
                response_size = .35
                response_halign = "center"
            elif len(response) < 21:
                response_size = .48
                response_halign = "center"
            elif len(response) < 26:
                response_size = .51
                response_halign = "center"
            else:
                response_size = .67
                response_halign = "left"

            chat_screen = screen_manager.get_screen('chat')
            chat_list = chat_screen.chat_list
            chat_list.add_widget(Response(text=response, size_hint_x=response_size, halign=response_halign))
            text_input.text = ""


if __name__ =="__main__":
    ChatBot().run()
