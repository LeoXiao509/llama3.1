import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session
from backend import kb, chat_engine  # 後端邏輯模組

# 前端界面代碼

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # 換成你自己的安全隨機字串

# 全域儲存所有會話 (單一使用者 demo)
sessions = {}

def get_active_session():
    sid = session.get('active_session_id')
    if sid in sessions:
        return sid
    # 如果沒有，就新建一個
    new_id = str(uuid.uuid4())
    sessions[new_id] = {'name': '新對話', 'messages': []}
    session['active_session_id'] = new_id
    return new_id

@app.route('/', methods=['GET', 'POST'])
def index():
    active_id = get_active_session()
    search_results = None

    if request.method == 'POST':
        if 'upload' in request.form:
            uploaded_files = request.files.getlist('files')
            if not uploaded_files:
                flash("未選擇任何檔案", "warning")
            else:
                # 添加调试信息
                print(f"收到檔案上傳: {[f.filename for f in uploaded_files]}")
                for file in uploaded_files:
                    print(f"檔案資訊: {file.filename}, 大小: {len(file.read())} bytes")
                    file.seek(0)  # 重置文件指针
                
                added_count = kb.add_files(uploaded_files)
                print(f"知識庫當前狀態 - 總條目: {len(kb.contexts)}, 已處理檔案: {len(kb.processed_files)}")
                
                if added_count > 0:
                    flash(f"成功新增 {added_count} 筆技術資料！", "success")
                else:
                    flash("未新增任何資料（可能是重複檔案或格式錯誤）", "warning")
            return redirect(url_for('index'))
        # -- 提交查詢
        if 'query' in request.form:
            query = request.form['query']
            sessions[active_id]['messages'].append({'role': 'user', 'content': query})
            search_results = kb.semantic_search(query)
            if not search_results:
                answer = "⚠️ 未找到相關技術資料，請先上傳技術文件。"
            else:
                answer = chat_engine.generate_response(query, search_results)
            sessions[active_id]['messages'].append({'role': 'assistant', 'content': answer})
            return redirect(url_for('index'))

    # 處理切換會話 (GET 參數)
    sid = request.args.get('session_id')
    if sid in sessions and sid != active_id:
        session['active_session_id'] = sid
        active_id = sid

    # 傳到 Template 的資料
    chat_history = sessions[active_id]['messages']
    kb_stats = {
        'total': len(kb.contexts),
        'files': len(kb.processed_files)
    }
    logo_url = url_for('static', filename='goatlogo.png')

    return render_template('index.html',
                           sessions=sessions,
                           active_session_id=active_id,
                           chat_history=chat_history,
                           kb_stats=kb_stats,
                           search_results=search_results,
                           logo_url=logo_url)

@app.route('/new_session')
def new_session():
    new_id = str(uuid.uuid4())
    sessions[new_id] = {'name': '新對話', 'messages': []}
    session['active_session_id'] = new_id
    return redirect(url_for('index'))


@app.route('/load_session')
def load_session():
    sid = request.args.get('session_id')
    # 確認這個 sid 存在於 sessions
    if sid in sessions:
        session['active_session_id'] = sid
    return redirect(url_for('index'))


@app.route('/rename_session', methods=['POST'])
def rename_session():
    sid = request.form['session_id']
    new_name = request.form['new_name'].strip()
    if sid in sessions and new_name:
        sessions[sid]['name'] = new_name
        session['active_session_id'] = sid
        flash("已修改對話名稱！", "success")
    return redirect(url_for('index'))

@app.route('/delete_session', methods=['POST'])
def delete_session():
    sid = request.form['session_id']
    if sid in sessions:
        sessions.pop(sid)
        flash("已刪除對話！", "warning")
        # 如果刪的是當前 active，則切到第一個或新建一個
        if session.get('active_session_id') == sid:
            if sessions:
                session['active_session_id'] = next(iter(sessions))
            else:
                new_id = str(uuid.uuid4())
                sessions[new_id] = {'name': '新對話', 'messages': []}
                session['active_session_id'] = new_id
    return redirect(url_for('index'))

# 先保留的兩個按鈕占位
@app.route('/goat_profile')
def goat_profile():
    flash("山羊履歷管理模組 功能尚未開放", "info")
    return redirect(url_for('index'))

@app.route('/sensor_module')
def sensor_module():
    flash("羊舍感測模組 功能尚未開放", "info")
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
