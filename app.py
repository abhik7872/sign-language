from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():  # put application's code here
    if request.method == "POST":
        room_id = request.form['room_id']
        return redirect(url_for("checkpoint", room_id=room_id))
    return render_template('index.html')


# @app.route("/room/<string:room_id>/")
# def enter_room(room_id):
#     if room_id not in session:
#         return redirect(url_for("checkpoint", room_id=room_id))
#     return render_template("room.html", room_id=room_id, display_name=session[room_id]["name"],
#                            mute_audio=session[room_id]["mute_audio"], mute_video=session[room_id]["mute_video"])
#
#
@app.route("/room/<string:room_id>/checkpoint/", methods=["GET", "POST"])
def entry_checkpoint(room_id):
    if request.method == "POST":
        display_name = request.form['display_name']
        mute_audio = request.form['mute_audio']
        mute_video = request.form['mute_video']
        session[room_id] = {"name": display_name, "mute_audio": mute_audio, "mute_video": mute_video}
        return redirect(url_for("enter_room", room_id=room_id))
    return render_template("checkpoint.html", room_id=room_id)


if __name__ == '__main__':
    app.run()
