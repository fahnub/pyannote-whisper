from flask import Flask, render_template, request, jsonify
from datetime import timedelta
from pyannote.audio import Pipeline
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from pydub import AudioSegment
import subprocess
import requests
import whisper
import shutil
import yt_dlp
import torch
import glob
import json
import re
import os

load_dotenv()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stt_model = whisper.load_model('small', device=device)
# openai.api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI()

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'postprocessor_args': [
        '-ar', '16000'
    ],
    'prefer_ffmpeg': True,
}

def get_id(url):
    pattern = r'(?:youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/watch\?v=|youtube\.com/user/.*/u/.*\?v=|vimeo\.com/\d+/video/|youtube\.googleusercontent\.com/|mp4|watch\?.*&v=|youtube\.com/\w+/.*(?:\?|&)v=|youtube\.com/(?:v|e|channel)/|youtube\.com/(?:v|e|user|c/|s)|youtube\.com/ytscreeningroom\?v=|youtube\.com/tv#|youtube\.com/leanback#|youtube\.com/dasmn#|youtube\.com/activate\?.*&v=|youtube\.com/e/|youtube\.com/live_playback\?v=)([^#\&\?]*).*'
    match = re.search(pattern, url)
    return match.group(1)

def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

def timeStr(t):
  return '{0:02d}:{1:02d}:{2:06.2f}'.format(round(t // 3600), round(t % 3600 // 60), t % 60)

def convert_to_wav(filename):
    command = ['ffmpeg', '-i', filename, '-ac', '1', '-ab', '256000', '-ar', '16000', f'{filename}.wav']
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    command = ['mv', f'{filename}.wav', 'audio/']
    subprocess.run(command)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    min_spk = 2
    max_spk = 26
    
    youtube_url = request.form.get('url')
    non_youtube_url = request.form.get('non_youtube_url')
    uploaded_file = request.files.get('file')
    num_speakers = request.form.get('numSpeakers')
    translate = request.form.get('translate')
    
    if num_speakers:
        try:
            num_speakers = int(num_speakers)
            if num_speakers < min_spk or num_speakers > max_spk:
                return jsonify({"Error": "Number of speakers must be between 2 and 26"})
            else:
                min_spk = num_speakers
                max_spk = num_speakers
        except ValueError:
            return jsonify({"Error": "Invalid input for number of speakers"})

    if youtube_url:    
        video_id = get_id(youtube_url)
        video_url = "https://www.youtube.com/watch?v=" + str(video_id)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            info_dict = ydl.extract_info(video_url, download=False)
            video_title = info_dict.get('title', None)

        filename = [f for f in glob.glob(f'*{video_id}].wav')][0].replace('/', '\\')

    if non_youtube_url:
        file = requests.get(non_youtube_url)
        audio_title = non_youtube_url
        filename = non_youtube_url.replace('/', '\\')
        with open(filename, 'wb') as f:
            f.write(file.content)
    
    if uploaded_file:
        for filename, handle in request.files.items():
            audio_title = filename
            temp = NamedTemporaryFile()
            handle.save(temp)
            convert_to_wav(temp.name)
    
        filename = temp.name.split('/')[-1] + ".wav"
    
    filepath = f"audio/{filename}"
    
    if youtube_url or non_youtube_url:
        shutil.move(filename, filepath)

    spacermilli = 2000
    silence = AudioSegment.silent(duration=spacermilli)
    audio = AudioSegment.from_wav(filepath)
    audio = silence.append(audio, crossfade=0)
    audio.export(filepath, format='wav')

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HUGGING_FACE_ACCESS_TOKEN"))
    pipeline.to(device)
    diarization = pipeline({'uri': 'filepath', 'audio': filepath}, min_speakers=min_spk, max_speakers=max_spk)

    diarization_file = f"diarization/{filename}.txt"
    
    with open(diarization_file, "w") as text_file:
        text_file.write(str(diarization))

    dzs = open(diarization_file).read().splitlines()

    groups = []
    g = []
    lastend = 0

    for d in dzs:   
        if g and (g[0].split()[-1] != d.split()[-1]):
            groups.append(g)
            g = []
        g.append(d)
  
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
        end = millisec(end)
        if (lastend > end):
            groups.append(g)
            g = [] 
        else:
            lastend = end
    if g:
        groups.append(g)

    audio = AudioSegment.from_wav(filepath)
    gidx = -1
    
    for g in groups:
        start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
        start = millisec(start)
        end = millisec(end)
        gidx += 1
        audio[start:end].export(str(gidx) + '.wav', format='wav')
    
    del pipeline, audio, diarization

    for i in range(len(groups)):
        audiof = str(i) + '.wav'
        result = stt_model.transcribe(audio=audiof, language='en', word_timestamps=True)
        with open(str(i)+'.json', "w") as outfile:
            json.dump(result, outfile, indent=4)
    
    speakers = {'SPEAKER_00':('Person A', '#e1ffc7', 'darkgreen'),
                'SPEAKER_01':('Person B', 'white', 'darkorange')}
    def_boxclr = 'white'
    def_spkrclr = 'orange'

    if youtube_url:
        preS = '<!DOCTYPE html>\n<html lang="en">\n\n<head>\n\t<meta charset="UTF-8">\n\t<meta name="viewport" content="width=device-width, initial-scale=1.0">\n\t<meta http-equiv="X-UA-Compatible" content="ie=edge">\n\t<title>' + video_title + '</title>\n\t<style>\n\t\tbody {\n\t\t\tfont-family: sans-serif;\n\t\t\tfont-size: 14px;\n\t\t\tcolor: #111;\n\t\t\tpadding: 0 0 1em 0;\n\t\t\tbackground-color: #efe7dd;\n\t\t}\n\n\t\ttable {\n\t\t\tborder-spacing: 10px;\n\t\t}\n\n\t\tth {\n\t\t\ttext-align: left;\n\t\t}\n\n\t\t.lt {\n\t\t\tcolor: inherit;\n\t\t\ttext-decoration: inherit;\n\t\t}\n\n\t\t.l {\n\t\t\tcolor: #050;\n\t\t}\n\n\t\t.s {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t.c {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t.e {\n\t\t\t/*background-color: white; Changing background color */\n\t\t\tborder-radius: 10px;\n\t\t\t/* Making border radius */\n\t\t\twidth: 50%;\n\t\t\t/* Making auto-sizable width */\n\t\t\tpadding: 0 0 0 0;\n\t\t\t/* Making space around letters */\n\t\t\tfont-size: 14px;\n\t\t\t/* Changing font size */\n\t\t\tmargin-bottom: 0;\n\t\t}\n\n\t\t.t {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t#player-div {\n\t\t\tposition: sticky;\n\t\t\ttop: 20px;\n\t\t\tfloat: right;\n\t\t\twidth: 40%\n\t\t}\n\n\t\t#player {\n\t\t\taspect-ratio: 16 / 9;\n\t\t\twidth: 100%;\n\t\t\theight: auto;\n\n\t\t}\n\n\t\ta {\n\t\t\tdisplay: inline;\n\t\t}\n\t</style>\n\t<script>\n\t\tvar tag = document.createElement(\'script\');\n\t\ttag.src = "https://www.youtube.com/iframe_api";\n\t\tvar firstScriptTag = document.getElementsByTagName(\'script\')[0];\n\t\tfirstScriptTag.parentNode.insertBefore(tag, firstScriptTag);\n\t\tvar player;\n\t\tfunction onYouTubeIframeAPIReady() {\n\t\t\tplayer = new YT.Player(\'player\', {\n\t\t\t\t//height: \'210\',\n\t\t\t\t//width: \'340\',\n\t\t\t\tvideoId: \''+ video_id + '\',\n\t\t\t});\n\n\n\n\t\t\t// This is the source "window" that will emit the events.\n\t\t\tvar iframeWindow = player.getIframe().contentWindow;\n\t\t\tvar lastword = null;\n\n\t\t\t// So we can compare against new updates.\n\t\t\tvar lastTimeUpdate = "-1";\n\n\t\t\t// Listen to events triggered by postMessage,\n\t\t\t// this is how different windows in a browser\n\t\t\t// (such as a popup or iFrame) can communicate.\n\t\t\t// See: https://developer.mozilla.org/en-US/docs/Web/API/Window/postMessage\n\t\t\twindow.addEventListener("message", function (event) {\n\t\t\t\t// Check that the event was sent from the YouTube IFrame.\n\t\t\t\tif (event.source === iframeWindow) {\n\t\t\t\t\tvar data = JSON.parse(event.data);\n\n\t\t\t\t\t// The "infoDelivery" event is used by YT to transmit any\n\t\t\t\t\t// kind of information change in the player,\n\t\t\t\t\t// such as the current time or a playback quality change.\n\t\t\t\t\tif (\n\t\t\t\t\t\tdata.event === "infoDelivery" &&\n\t\t\t\t\t\tdata.info &&\n\t\t\t\t\t\tdata.info.currentTime\n\t\t\t\t\t) {\n\t\t\t\t\t\t// currentTime is emitted very frequently (milliseconds),\n\t\t\t\t\t\t// but we only care about whole second changes.\n\t\t\t\t\t\tvar ts = (data.info.currentTime).toFixed(1).toString();\n\t\t\t\t\t\tts = (Math.round((data.info.currentTime) * 5) / 5).toFixed(1);\n\t\t\t\t\t\tts = ts.toString();\n\t\t\t\t\t\tconsole.log(ts)\n\t\t\t\t\t\tif (ts !== lastTimeUpdate) {\n\t\t\t\t\t\t\tlastTimeUpdate = ts;\n\n\t\t\t\t\t\t\t// It\'s now up to you to format the time.\n\t\t\t\t\t\t\t//document.getElementById("time2").innerHTML = time;\n\t\t\t\t\t\t\tword = document.getElementById(ts)\n\t\t\t\t\t\t\tif (word) {\n\t\t\t\t\t\t\t\tif (lastword) {\n\t\t\t\t\t\t\t\t\tlastword.style.fontWeight = \'normal\';\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t\tlastword = word;\n\t\t\t\t\t\t\t\t//word.style.textDecoration = \'underline\';\n\t\t\t\t\t\t\t\tword.style.fontWeight = \'bold\';\n\n\t\t\t\t\t\t\t\tlet toggle = document.getElementById("autoscroll");\n\t\t\t\t\t\t\t\tif (toggle.checked) {\n\t\t\t\t\t\t\t\t\tlet position = word.offsetTop - 20;\n\t\t\t\t\t\t\t\t\twindow.scrollTo({\n\t\t\t\t\t\t\t\t\t\ttop: position,\n\t\t\t\t\t\t\t\t\t\tbehavior: \'smooth\'\n\t\t\t\t\t\t\t\t\t});\n\t\t\t\t\t\t\t\t}\n\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t})\n\t\t}\n\t\tfunction jumptoTime(timepoint, id) {\n\t\t\tevent.preventDefault();\n\t\t\thistory.pushState(null, null, "#" + id);\n\t\t\tplayer.seekTo(timepoint);\n\t\t\tplayer.playVideo();\n\t\t}\n\t</script>\n</head>\n\n<body>\n\t<h2>' + video_title + '</h2>\n\t<i>Click on a part of the transcription, to jump to its video, and get an anchor to it in the address\n\t\tbar<br><br></i>\n\t<div id="player-div">\n\t\t<div id="player"></div>\n\t\t<div><label for="autoscroll">auto-scroll: </label>\n\t\t\t<input type="checkbox" id="autoscroll" checked>\n\t\t</div>\n\t</div>\n  '
    else:
        preS = '\n<!DOCTYPE html>\n<html lang="en">\n\n<head>\n\t<meta charset="UTF-8">\n\t<meta name="viewport" content="whtmlidth=device-width, initial-scale=1.0">\n\t<meta http-equiv="X-UA-Compatible" content="ie=edge">\n\t<title>' + audio_title + '</title>\n\t<style>\n\t\tbody {\n\t\t\tfont-family: sans-serif;\n\t\t\tfont-size: 14px;\n\t\t\tcolor: #111;\n\t\t\tpadding: 0 0 1em 0;\n\t\t\tbackground-color: #efe7dd;\n\t\t}\n\n\t\ttable {\n\t\t\tborder-spacing: 10px;\n\t\t}\n\n\t\tth {\n\t\t\ttext-align: left;\n\t\t}\n\n\t\t.lt {\n\t\t\tcolor: inherit;\n\t\t\ttext-decoration: inherit;\n\t\t}\n\n\t\t.l {\n\t\t\tcolor: #050;\n\t\t}\n\n\t\t.s {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t.c {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t.e {\n\t\t\t/*background-color: white; Changing background color */\n\t\t\tborder-radius: 10px;\n\t\t\t/* Making border radius */\n\t\t\twidth: 50%;\n\t\t\t/* Making auto-sizable width */\n\t\t\tpadding: 0 0 0 0;\n\t\t\t/* Making space around letters */\n\t\t\tfont-size: 14px;\n\t\t\t/* Changing font size */\n\t\t\tmargin-bottom: 0;\n\t\t}\n\n\t\t.t {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t#player-div {\n\t\t\tposition: sticky;\n\t\t\ttop: 20px;\n\t\t\tfloat: right;\n\t\t\twidth: 40%\n\t\t}\n\n\t\t#player {\n\t\t\taspect-ratio: 16 / 9;\n\t\t\twidth: 100%;\n\t\t\theight: auto;\n\t\t}\n\n\t\ta {\n\t\t\tdisplay: inline;\n\t\t}\n\t</style>';
        preS += '\n\t<script>\n\twindow.onload = function () {\n\t\t\tvar player = document.getElementById("audio_player");\n\t\t\tvar player;\n\t\t\tvar lastword = null;\n\n\t\t\t// So we can compare against new updates.\n\t\t\tvar lastTimeUpdate = "-1";\n\n\t\t\tsetInterval(function () {\n\t\t\t\t// currentTime is checked very frequently (1 millisecond),\n\t\t\t\t// but we only care about whole second changes.\n\t\t\t\tvar ts = (player.currentTime).toFixed(1).toString();\n\t\t\t\tts = (Math.round((player.currentTime) * 5) / 5).toFixed(1);\n\t\t\t\tts = ts.toString();\n\t\t\t\tconsole.log(ts);\n\t\t\t\tif (ts !== lastTimeUpdate) {\n\t\t\t\t\tlastTimeUpdate = ts;\n\n\t\t\t\t\t// Its now up to you to format the time.\n\t\t\t\t\tword = document.getElementById(ts)\n\t\t\t\t\tif (word) {\n\t\t\t\t\t\tif (lastword) {\n\t\t\t\t\t\t\tlastword.style.fontWeight = "normal";\n\t\t\t\t\t\t}\n\t\t\t\t\t\tlastword = word;\n\t\t\t\t\t\t//word.style.textDecoration = "underline";\n\t\t\t\t\t\tword.style.fontWeight = "bold";\n\n\t\t\t\t\t\tlet toggle = document.getElementById("autoscroll");\n\t\t\t\t\t\tif (toggle.checked) {\n\t\t\t\t\t\t\tlet position = word.offsetTop - 20;\n\t\t\t\t\t\t\twindow.scrollTo({\n\t\t\t\t\t\t\t\ttop: position,\n\t\t\t\t\t\t\t\tbehavior: "smooth"\n\t\t\t\t\t\t\t});\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}, 0.1);\n\t\t}\n\n\t\tfunction jumptoTime(timepoint, id) {\n\t\t\tvar player = document.getElementById("audio_player");\n\t\t\thistory.pushState(null, null, "#" + id);\n\t\t\tplayer.pause();\n\t\t\tplayer.currentTime = timepoint;\n\t\t\tplayer.play();\n\t\t}\n\t\t</script>\n\t</head>';
        preS += '\n\n<body>\n\t<h2>' + audio_title + '</h2>\n\t<i>Click on a part of the transcription, to jump to its portion of audio, and get an anchor to it in the address\n\t\tbar<br><br></i>\n\t<div id="player-div">\n\t\t<div id="player">\n\t\t\t<audio controls="controls" id="audio_player">\n\t\t\t\t<source src="input.wav" />\n\t\t\t</audio>\n\t\t</div>\n\t\t<div><label for="autoscroll">auto-scroll: </label>\n\t\t\t<input type="checkbox" id="autoscroll" checked>\n\t\t</div>\n\t</div>\n';

    postS = '\t</body>\n</html>'

    html = list(preS)
    txt = list("")
    gidx = -1
    
    for g in groups:  
        shift = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
        shift = millisec(shift) - spacermilli
        shift=max(shift, 0)
  
        gidx += 1
  
        captions = json.load(open(str(gidx) + '.json'))['segments']

        if captions:
            speaker = g[0].split()[-1]
            boxclr = def_boxclr
            spkrclr = def_spkrclr
            if speaker in speakers:
                speaker, boxclr, spkrclr = speakers[speaker] 
            
            html.append(f'<div class="e" style="background-color: {boxclr}">\n');
            html.append('<p  style="margin:0;padding: 5px 10px 10px 10px;word-wrap:normal;white-space:normal;">\n')
            html.append(f'<span style="color:{spkrclr};font-weight: bold;">{speaker}</span><br>\n\t\t\t\t')
            
            for c in captions:
                start = shift + c['start'] * 1000.0 
                start = start / 1000.0
                end = (shift + c['end'] * 1000.0) / 1000.0      
                txt.append(f'[{timeStr(start)} --> {timeStr(end)}] [{speaker}] {c["text"]}\n')

                for i, w in enumerate(c['words']):
                    if w == "":
                        continue
                    start = (shift + w['start']*1000.0) / 1000.0        
                    html.append(f'<a href="#{timeStr(start)}" id="{"{:.1f}".format(round(start*5)/5)}" class="lt" onclick="jumptoTime({int(start)}, this.id)">{w["word"]}</a><!--\n\t\t\t\t-->')
            html.append('</p>\n')
            html.append(f'</div>\n')

    html.append(postS)

    with open(f"capspeaker.txt", "w", encoding='utf-8') as file:
        s = "".join(txt)
        file.write(s)
        print('captions saved to capspeaker.txt:')
        print(s+'\n')

    with open(f"capspeaker.html", "w", encoding='utf-8') as file:
        s = "".join(html)
        file.write(s)
        print('captions saved to capspeaker.html:')
        print(s+'\n')

    return jsonify({"Success": "Diarization and Transcription Complete. HTML Saved."})

if __name__ == '__main__':
    app.run(debug=True)
