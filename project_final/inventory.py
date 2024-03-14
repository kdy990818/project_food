from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pymysql as pymysql
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import re
# from .recommended_foods import get_recommendations

app = Flask(__name__)
app.secret_key = 'your_secret_key'

df = pd.read_excel('C:\\Users\\nanzz\\Desktop\\프로젝트\\구매테이블_수정.xlsx')
#nut = pd.read_excel('C:\\Users\\korea\\Desktop\\재고관리자료\\프로젝트자료\\주피터\\추천시스템\\구매테이블_수정.xlsx')
df['menu'] = df['menu'].str.replace(' ','')
df['amount'] = df['amount'].str.replace(' ','')

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/get_menu_list', methods=['GET'])
def get_menu_list():
    term = request.args.get('term', '')
    matched_menus = df[df['menu'].str.contains(term)]['menu'].tolist()[:5] # 여기에서 자동 완성 
    return jsonify(matched_menus)

@app.route('/menuSearch', methods=['GET'])
def menuSearch():
    
    return render_template('menuSearch.html')

@app.route('/purchase', methods=['GET', 'POST'])
def purchase():
    result_text = None
    filtered_df = None

    if request.method == 'POST':
        menu_input = request.form['menu']
        multiplier = int(request.form['multiplier'])  # 입력한 인원 수 가져오기

        # 입력 받은 'menu' 값과 일치하는 행을 필터링합니다.
        filtered_df = df[df['menu'] == menu_input]

        if not filtered_df.empty:
            menu = filtered_df['menu'].iloc[0]

            # amount = filtered_df['amount'].iloc[0]
            result_text = f"{menu} {multiplier}인분"

            # 필터링된 결과에서 'menu', 'rcp', 'amount', 'category' 열을 보여줍니다.
            display_columns = ['menu', 'rcp', 'amount', 'category']
            filtered_df['rcp'] = filtered_df['rcp'].apply(lambda x: x.split(', '))
            # filtered_df['amount'] = filtered_df['amount'].apply(lambda x: ', '.join([str(float(amount) * multiplier) for amount in x.split(', ')]))
            filtered_df = filtered_df[display_columns].reset_index(drop=True)

    return render_template('purchase.html',  result_text=result_text)

@app.route('/inventory', methods=['GET', 'POST'])
def inventory():
    # 데이터프레임 생성 (주어진 데이터를 그대로 사용)
    df = pd.read_excel('C:\\Users\\nanzz\\Desktop\\프로젝트\\구매테이블_수정.xlsx')
    df['rcp'] = df['rcp'].apply(lambda x: str(x).split(','))
    matched_menus = []  # 초기값 설정
    no_results_message = ""  # 초기값 설정

    if request.method == 'POST':
        user_input = request.form.get('ingredients').split()
        filter_by = request.form.get('filter_by')

        for index, row in df.iterrows():
            if any(ingredient.lower() in [item.strip().lower() for item in row['rcp']] for ingredient in user_input):
                if not filter_by or row['code_nm'].startswith(filter_by):
                    matched_menus.append(row['menu'])

        if user_input and not matched_menus:
            no_results_message = "입력한 식재료에 해당하는 메뉴가 없습니다."
        
    return render_template('inventory.html', matched_menus=matched_menus, no_results_message=no_results_message)


@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    menu_text = request.form.get('menuText')
    if 'cart' not in session:
        session['cart'] = []

   # 세션에 저장된 값을 불러와서 새로운 메뉴를 추가합니다.
    cart = session.get('cart', [])
    cart.append(menu_text)
    session['cart'] = cart
    return jsonify({"message": "추가되었습니다."})


@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    menu_text = request.form.get('menuText')
    cart = session.get('cart', [])

    if menu_text in cart:
        cart.remove(menu_text)
        session['cart'] = cart

        # 세션에서 모든 장바구니 정보 삭제
        session.pop('cart', None)

        return jsonify({"message": "삭제되었습니다."})
    else:
        return jsonify({"message": "해당 메뉴가 장바구니에 없습니다."})


@app.route('/get_recipe_info', methods=['POST'])
def get_recipe_info():
    cart_items = request.form.getlist('menuItems[]')
    rcp_info = []
    # multiplier = int(request.form['multiplier'])  # 입력한 인원 수 가져오기

    for item in cart_items:
        filtered_row = df[df['menu'] == item]
        if not filtered_row.empty:
            rcp_info.append({
                'menu': filtered_row['menu'].iloc[0],
                'rcp': filtered_row['rcp'].iloc[0],
                'amount': filtered_row['amount'].iloc[0]
            })

    return jsonify({"rcp_info": rcp_info})


@app.route('/menu',methods=['GET', 'POST'])
def menu():
    if request.method == 'POST':

        conn = pymysql.connect(host = '127.0.0.1',
                            user = 'root',
                            password = '1234',
                            db = 'food',
                            charset = 'utf8'
                            )

        cur1 = conn.cursor()
        cur2 = conn.cursor()

        sql_nut = 'select * from nut_table'
        cur1.execute(sql_nut) #질의 연결
        rows1 = cur1.fetchall()

        #칼럼명 추출하기
        col_names1 = cur1.description
        col_name1 = []

        # 칼럼명에 따라 하나씩 저장
        for colname in col_names1:
            col_name1.append(colname[0])


        # 데이터 프레임으로 변경
        foods = pd.DataFrame(rows1, columns = col_name1)

        code_to_category = {
            'RC' : 'RC',
            'SP' : 'SP',
            'SB' : 'SB',
            'MN' : 'MN',
            'DS' : 'DS',
            'KM' : 'KM'
        }

        nut_avg1 = {
            'eng' : (680.0, 880.0),'car' : (82.2, 112.2),'pro' : (11.5, 21.5),'fat' : (14.2, 18.2)
        }
        nut_avg2 = {
            'eng' : (940.0, 1140.0),'car' : (114.6, 144.6),'pro' : (17.0, 27.0),'fat' : (19.6, 23.6)
        }
        nut_avg3 = {
            'eng' : (680.0, 880.0),'car' : (82.2, 112.2),'pro' : (11.5, 21.5),'fat' : (14.2, 18.2)
        }


        sim = {
            'eng' : 10.0,
            'car' : 5.3,
            'pro' : 0.3,
            'fat' : 0.25
        }

        sim_df = pd.DataFrame.from_dict(sim, orient='index', columns=['Value'])

        foods['code_cate'] = foods['code_nm'].str[:2].map(code_to_category)

        #code_nm의 앞 두글자가 MN인 행의 category 열 값을 메인메뉴로 바꾼 것 적용
        foods.loc[foods['code_nm'].str[:2] == 'MN', 'category'] = '메인메뉴'

        #밥은 누룽지와 다른 메뉴와 궁합이 좋은 밥들로만 다시 저장
        foods = foods[(foods['code_nm'].str[:2].isin(['SP', 'MN', 'SB', 'DS', 'KM', 'SL'])) |
                        (foods['code_nm'].str[:2] == 'RC') &
                        (foods['menu'].isin(['현미밥','흑미밥','차조밥','율무밥','쌀밥','오곡밥','보리밥','귀리밥','기장밥','수수밥','잡곡밥', '누룽지']))]


        def recommend_foods_for_keyword_mo(keyword, keyword_indices, num_recommendations, sb_bokkeum_added, sb_jorim_added, sb_na_added, used_foods, cosine_similarities):

            for keyword_index in keyword_indices:
                if len(recommended_foods[keyword]) >= num_recommendations:
                    break

                cluster_index = cluster_indices[keyword_index]
                similar_indices = cosine_similarities[:, cluster_index].argsort()[::-1]
                for _ in range(1):
                    for j in similar_indices:
                        if keyword == 'SB':
                            if foods.iloc[j]['code_cate'] == 'SB' and foods.iloc[j]['category']  in ['생채·무침류', '나물·숙채류'] and not sb_bokkeum_added:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.2 * cosine_similarity_score + 0.8 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                sb_bokkeum_added = True
                                break
                            elif foods.iloc[j]['code_cate'] == 'SB' and foods.iloc[j]['category'] in ['조림류', '볶음류'] and not sb_jorim_added:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                sb_jorim_added = True
                                break
                            elif foods.iloc[j]['code_cate'] == 'SB' and foods.iloc[j]['category'] not in ['조림류', '볶음류', '구이류','튀김류', '서브메뉴3'] and not sb_na_added:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                sb_na_added = True
                                break
                        elif keyword == 'RC':
                            if foods.iloc[j]['code_cate'] == 'RC' and foods.iloc[j]['menu'] in ['현미밥', '흑미밥',  '차조밥', '율무밥', '쌀밥', '오곡밥', '보리밥', '귀리밥',  '기장밥', '수수밥',  '잡곡밥']:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.8 * cosine_similarity_score + 0.2 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                break

                        elif keyword == 'DS':
                            if foods.iloc[j]['menu'] == '누룽지':
                                food_row = foods[foods['menu'] == '누룽지'].iloc[0]
                                keyword_similarity_score = calculate_keyword_similarity(keyword, food_row)
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                        elif keyword == 'MN':
                            if foods.iloc[j]['code_cate'] == 'MN' and foods.iloc[j]['car'] <= 25 and foods.iloc[j]['pro'] <= 8.0:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,recommended_foods, used_foods)
                                break
                        else:
                            if foods.iloc[j]['code_cate'] == keyword and foods.iloc[j]['eng'] <= 200:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.9 * cosine_similarity_score + 0.1 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                break

        def recommended_keywords1(keywords, cosine_similarities):
            for keyword in keywords:
                keyword_indices = list(foods[foods['code_cate'] == keyword].index)
                num_recommendations = 1 if keyword != 'SB' else 3
                sb_bokkeum_added = False
                sb_jorim_added = False
                sb_na_added = False
                used_foods = set()
                recommend_foods_for_keyword_mo(keyword, keyword_indices, num_recommendations, sb_bokkeum_added, sb_jorim_added, sb_na_added, used_foods, cosine_similarities)

        def recommend_foods_for_keyword_la(keyword, keyword_indices, num_recommendations, sb_bokkeum_added, sb_jorim_added, sb_na_added, used_foods, cosine_similarities):

            for keyword_index in keyword_indices:
                if len(recommended_foods[keyword]) >= num_recommendations:
                    break

                cluster_index = cluster_indices[keyword_index]
                similar_indices = np.argsort(cosine_similarities[:, cluster_index])
                for _ in range(1):
                    for j in similar_indices:
                        if keyword == 'SB':
                            if foods.iloc[j]['code_cate'] == 'SB' and foods.iloc[j]['category'] == '볶음류' and foods.iloc[j]['pro'] <= 8.0 and not sb_bokkeum_added:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                sb_bokkeum_added = True
                                break
                            elif foods.iloc[j]['code_cate'] == 'SB' and foods.iloc[j]['category'] in ['조림류', '구이류', '튀김류'] and foods.iloc[j]['eng'] <= 50.0 and foods.iloc[j]['pro'] <= 8.0 and not sb_jorim_added:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.2 * cosine_similarity_score + 0.8 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                sb_jorim_added = True
                                break
                            elif foods.iloc[j]['code_cate'] == 'SB' and foods.iloc[j]['category'] not in ['조림류', '볶음류', '구이류', '튀김류', '서브메뉴3'] and foods.iloc[j]['pro'] < 10 and foods.iloc[j]['car'] < 10 and not sb_na_added:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.2 * cosine_similarity_score + 0.8 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                sb_na_added = True
                                break
                        elif keyword == 'RC':
                            if foods.iloc[j]['code_cate'] == 'RC' and foods.iloc[j]['menu'] in ['현미밥', '흑미밥',  '차조밥', '율무밥', '쌀밥', '오곡밥', '보리밥', '귀리밥',  '기장밥', '수수밥',  '잡곡밥']:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.3 * cosine_similarity_score + 0.7 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                break
                        elif keyword == 'MN':
                            if foods.iloc[j]['code_cate'] == 'MN' and foods.iloc[j]['car'] <= 30 and foods.iloc[j]['pro'] <= 20:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                break
                        else:
                            if foods.iloc[j]['code_cate'] == keyword and foods.iloc[j]['pro'] <= 10:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                break

        def recommended_keywords2(keywords, cosine_similarities):
            for keyword in keywords:
                keyword_indices = list(foods[foods['code_cate'] == keyword].index)
                num_recommendations = 1 if keyword != 'SB' else 3
                sb_bokkeum_added = False
                sb_jorim_added = False
                sb_na_added = False
                used_foods = set()
                recommend_foods_for_keyword_la(keyword, keyword_indices, num_recommendations, sb_bokkeum_added, sb_jorim_added, sb_na_added, used_foods, cosine_similarities)

        def recommend_foods_for_keyword_di(keyword, keyword_indices, num_recommendations, sb_bokkeum_added, sb_jorim_added, sb_na_added,sb_na1_added, used_foods, cosine_similarities):

            for keyword_index in keyword_indices:
                if len(recommended_foods[keyword]) >= num_recommendations:
                    break

                cluster_index = cluster_indices[keyword_index]
                similar_indices = cosine_similarities[:, cluster_index].argsort()[::-5]

                for _ in range(1):
                    for j in similar_indices:
                        if keyword == 'SB':
                            if foods.iloc[j]['code_cate'] == 'SB' and foods.iloc[j]['category'] in ['생채·무침류', '나물·숙채류'] and foods.iloc[j]['pro'] <= 10.0 and not sb_bokkeum_added:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                sb_bokkeum_added = True
                                break
                            elif foods.iloc[j]['code_cate'] == 'SB' and foods.iloc[j]['category'] in ['조림류', '볶음류'] and foods.iloc[j]['pro'] <= 8.0 and not sb_jorim_added:
                                    keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                    cosine_similarity_score = cosine_similarities[j, cluster_index]
                                    combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                    add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                    recommended_foods, used_foods)
                                    sb_jorim_added = True
                                    break
                            elif foods.iloc[j]['code_cate'] == 'SB' and foods.iloc[j]['category'] not in ['조림류', '볶음류', '구이류', '서브메뉴3'] and foods.iloc[j]['pro'] <= 9.0 and not sb_na_added:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,
                                                                recommended_foods, used_foods)
                                sb_na_added = True
                                break
                            elif foods.iloc[j]['code_cate'] == 'SB' and foods.iloc[j]['category'] not in ['조림류', '볶음류', '튀김류'] and foods.iloc[j]['car'] <= 12.0 and foods.iloc[j]['pro'] <= 8.0 and not sb_na1_added:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,recommended_foods, used_foods)
                                sb_na1_added = True
                                break
                        elif keyword == 'RC':
                            if foods.iloc[j]['code_cate'] == 'RC' and foods.iloc[j]['menu'] in ['현미밥', '흑미밥',  '차조밥', '율무밥', '쌀밥', '오곡밥', '보리밥', '귀리밥',  '기장밥', '수수밥',  '잡곡밥']:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,recommended_foods, used_foods)
                                break
                        elif keyword == 'MN':
                            if foods.iloc[j]['code_cate'] == 'MN' and foods.iloc[j]['car'] <= 25 and foods.iloc[j]['pro'] <= 8.0:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity,recommended_foods, used_foods)
                                break
                        else:
                            if foods.iloc[j]['code_cate'] == keyword and foods.iloc[j]['pro'] <= 10.0:
                                keyword_similarity_score = calculate_keyword_similarity(keyword, foods.iloc[j])
                                cosine_similarity_score = cosine_similarities[j, cluster_index]
                                combined_similarity = 0.7 * cosine_similarity_score + 0.3 * keyword_similarity_score
                                add_recommendation_with_similarity(keyword, foods.iloc[j], combined_similarity, recommended_foods, used_foods)
                                break


        def add_recommendation_with_similarity(keyword, food_row, similarity_score, recommended_foods, used_foods):
            recommended_food = food_row['menu']
            if recommended_food not in used_foods:
                recommended_cate = food_row['category']
                recommended_eng = food_row['eng']
                recommended_car = food_row['car']
                recommended_pro = food_row['pro']
                recommended_fat = food_row['fat']

                recommendations.append((keyword, recommended_food, recommended_eng, recommended_car,
                                        recommended_pro, recommended_fat, recommended_cate, similarity_score))
                recommended_foods[keyword].append(recommended_food)
                used_foods.add(recommended_food)


        def recommended_keywords3(keywords, cosine_similarities):
            for keyword in keywords:
                keyword_indices = list(foods[foods['code_cate'] == keyword].index)
                num_recommendations = 1 if keyword != 'SB' else 4
                sb_bokkeum_added = False
                sb_jorim_added = False
                sb_na_added = False
                sb_na1_added = False
                used_foods = set()
                recommend_foods_for_keyword_di(keyword, keyword_indices, num_recommendations, sb_bokkeum_added, sb_jorim_added, sb_na_added,sb_na1_added, used_foods, cosine_similarities)


        def calculate_keyword_similarity(keyword, food_row):
            keyword_weights = {
                'RC': {'eng': 0.2, 'car': 0.1, 'pro': 0.2, 'fat': 0.1},
                'SP': {'eng': 0.2, 'car': 0.3, 'pro': 0.1, 'fat': 0.1},
                'MN': {'eng': 0.3, 'car': 0.1, 'pro': 0.2, 'fat': 0.1},
                'SB': {'eng': 0.1, 'car': 0.2, 'pro': 0.1, 'fat': 0.1},
                'DS': {'eng': 0.1, 'car': 0.1, 'pro': 0.1, 'fat': 0.1},
                'KM': {'eng': 0.2, 'car': 0.1, 'pro': 0.1, 'fat': 0.1}
            }

            keyword_weights_for_food = keyword_weights[keyword]
            food_eng = food_row['eng']
            food_car = food_row['car']
            food_pro = food_row['pro']
            food_fat = food_row['fat']

            keyword_similarity_score = (
                    keyword_weights_for_food['eng'] * food_eng +
                    keyword_weights_for_food['car'] * food_car +
                    keyword_weights_for_food['pro'] * food_pro +
                    keyword_weights_for_food['fat'] * food_fat
            )
            return keyword_similarity_score


        # 유사도 값의 최댓값을 기준으로 정규화
        max_similarity_value = sim_df['Value'].max()
        sim_df['Value'] = sim_df['Value'] / max_similarity_value

        # 열 순서에 맞게 유사도 값을 리스트로 저장
        similarity_values = [sim_df.loc[column, 'Value'] for column in ['eng', 'car', 'pro', 'fat']]

        # 유사도 값을 합으로 나누어 정규화
        similarity_values_normalized = similarity_values / sum(similarity_values)

        # 정규화된 유사도 값을 원본 데이터프레임에 추가
        normalized_sim_values = pd.DataFrame([similarity_values_normalized], columns=['eng', 'car', 'pro', 'fat'])
        normalized_foods = pd.concat([foods, normalized_sim_values], axis=1)

        # TF-IDF 준비
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(normalized_foods[['eng', 'car', 'pro', 'fat']].astype(str).apply(' '.join, axis=1))

        # PCA를 사용하여 차원 축소
        pca = PCA(n_components=2)  # 2차원으로 축소
        reduced_tfidf_matrix = pca.fit_transform(tfidf_matrix.toarray())

        num_clusters = 8
        best_kmeans = None
        best_silhouette_score = -1

        # 군집화를 여러 번 반복하여 Silhouette Score 최대화
        for _ in range(10):
            kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
            cluster_indices = kmeans.fit_predict(reduced_tfidf_matrix)

            # Silhouette Score 계산
            silhouette_avg = silhouette_score(tfidf_matrix, cluster_indices)

            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_kmeans = kmeans

        # 최적의 군집화 결과
        final_cluster_indices = best_kmeans.labels_
        cluster_centers = best_kmeans.cluster_centers_
        cosine_similarities = cosine_similarity(reduced_tfidf_matrix, cluster_centers)
        recommendations = []

        # 실루엣 스코어 계산
        silhouette_avg = silhouette_score(tfidf_matrix, cluster_indices)
        print(f"Silhouette Score: {silhouette_avg}")

        sum_eng = sum_car = sum_pro = sum_fat = 0

        keywords1 = ['RC', 'SP', 'MN', 'SB', 'DS', 'KM']
        keywords2 = ['RC', 'SP', 'MN', 'SB', 'DS', 'KM']
        keywords3 = ['RC', 'SP', 'MN', 'SB', 'SL', 'KM']

        recommended_foods = {keyword: [] for keyword in keywords1}
        recommended_keywords1(keywords1, cosine_similarities)
        recommended_foods = {keyword: [] for keyword in keywords2}
        recommended_keywords2(keywords2, cosine_similarities)
        recommended_foods = {keyword: [] for keyword in keywords3}
        recommended_keywords3(keywords3, cosine_similarities)


        chunk_size = 8
        chunks = [recommendations[i:i + chunk_size] for i in range(0, len(recommendations), chunk_size)]
        # Display recommendations
        nut_avgs = [nut_avg1, nut_avg2, nut_avg3]


        final_result = "" #출력값 저장할 변수 지정 

        for chunk_idx, chunk in enumerate(chunks):
            print(f"Chunk {chunk_idx + 1}:")
            chunk_sum_eng = chunk_sum_car = chunk_sum_pro = chunk_sum_fat = 0

            chunk_result = f"Chunk {chunk_idx + 1}:\n" #추가

            for idx, rec in enumerate(chunk):
                chunk_sum_eng += rec[2]
                chunk_sum_car += rec[3]
                chunk_sum_pro += rec[4]
                chunk_sum_fat += rec[5]
                
                chunk_result += f"{idx + 1}. {rec[0]}: {rec[1]} (eng: {rec[2]}, 탄수화물: {rec[3]}, 단백질: {rec[4]}, 지방: {rec[5]}, 카테고리 : {rec[6]})\n" #추가

                print(
                    f"{idx + 1}. {rec[0]}: {rec[1]} (eng: {rec[2]}, 탄수화물: {rec[3]}, 단백질: {rec[4]}, 지방: {rec[5]}, 카테고리 : {rec[6]})")

            print(f"Chunk {chunk_idx + 1} 열량 : {chunk_sum_eng:.2f}, 탄수화물: {chunk_sum_car:.2f}, 단백질: {chunk_sum_pro:.2f}, 지방: {chunk_sum_fat:.2f}")
            current_nut_avg = nut_avgs[chunk_idx % len(nut_avgs)]

            diff_eng = chunk_sum_eng - (current_nut_avg['eng'][1] + current_nut_avg['eng'][0]) / 2
            diff_car = chunk_sum_car - (current_nut_avg['car'][1] + current_nut_avg['car'][0]) / 2
            diff_pro = chunk_sum_pro - (current_nut_avg['pro'][1] + current_nut_avg['pro'][0]) / 2
            diff_fat = chunk_sum_fat - (current_nut_avg['fat'][1] + current_nut_avg['fat'][0]) / 2

            final_result += chunk_result + "\n" #추가


            def format_with_sign_or_zero(diff, lower_limit, upper_limit):
                if lower_limit <= diff < upper_limit:
                    return "0.00"
                return f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"

            print(
                f"Differences: 열량 {format_with_sign_or_zero(diff_eng, current_nut_avg['eng'][0], current_nut_avg['eng'][1])} kcal, " \
                f"탄수화물 {format_with_sign_or_zero(diff_car, current_nut_avg['car'][0], current_nut_avg['car'][1])} g, " \
                f"단백질 {format_with_sign_or_zero(diff_pro, current_nut_avg['pro'][0], current_nut_avg['pro'][1])} g, " \
                f"지방 {format_with_sign_or_zero(diff_fat, current_nut_avg['fat'][0], current_nut_avg['fat'][1])} g")
            print()

            saved_result = final_result

            saved_result = re.sub(r'\([^)]*\)', '', saved_result) # 괄호 안 들어있는 내용 삭제
            saved_result = saved_result.replace("Chunk 1", "조식").replace("Chunk 2", "중식").replace("Chunk 3", "석식")
            saved_result = re.sub(r'\d+\.\s+[A-Z]+:', '', saved_result) # 1.. RC:와 같은 부분 삭제



            meals = saved_result.strip().split('\n')

            # 조식, 중식, 석식을 나누어 저장할 리스트 생성
            breakfast_menu = []
            lunch_menu = []
            dinner_menu = []

            current_meal = None  # 현재 처리 중인 식사 (조식, 중식, 석식)

            # 메뉴를 각 식사별로 나누어 저장
            for meal in meals:
                if meal.startswith("조식:"):
                    current_meal = breakfast_menu
                elif meal.startswith("중식:"):
                    current_meal = lunch_menu
                elif meal.startswith("석식:"):
                    current_meal = dinner_menu
                else:
                    current_meal.append(meal.strip())
            
            breakfast_menu = list(filter(None, breakfast_menu))
            lunch_menu = list(filter(None, lunch_menu))
            dinner_menu = list(filter(None, dinner_menu))


        return render_template('menu.html', breakfast_menu = breakfast_menu, lunch_menu = lunch_menu, dinner_menu = dinner_menu)
    return render_template('menu.html', breakfast_menu = '', lunch_menu = '', dinner_menu = '')

@app.route('/refresh', methods=['GET'])
def refresh():
    return redirect(url_for('menu'))

if __name__ == '__main__':
    app.run(debug=True)


