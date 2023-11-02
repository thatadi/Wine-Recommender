from flask import Flask , request , render_template ,request
import pickle
import pandas as pd
expensive_df = pickle.load(open('templates/expensive.pkl','rb'))
wine_pivot = pickle.load(open('templates/wine_pivot.pkl','rb'))
model_knn = pickle.load(open('templates/model_knn.pkl','rb'))
points_df = pickle.load(open('templates/points.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/main')
def main():
    return render_template("main.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/prediction')
def prediction():
   
   return render_template("prediction.html",  
                           variety=wine_pivot.index)


@app.route('/expensive')
def expensive():
    return render_template("expensive.html",
                           title=list(expensive_df['title'].values),
                           variety=list(expensive_df['variety'].values),
                           price=list(expensive_df['price'].values),
                           winery=list(expensive_df['winery'].values),
                           province=list(expensive_df['province'].values),
                           description=list(expensive_df['description'].values),
                           )

@app.route('/points')
def points():
    title_list = list(points_df['title'].values)
    variety_list = list(points_df['variety'].values)
    price_list = list(points_df['price'].values)
    point_list = list(points_df['points'].values)
    return render_template("points.html",name=title_list,
                           variety=variety_list,
                           price=price_list,
                           point=point_list)

@app.route('/recommendation', methods=['POST','GET'])
def recommendation():
    search_value = request.form.get('search-box')
    index_number=0
    for i in range (wine_pivot.shape[0]):
      if search_value==wine_pivot.index[i]:
        index_number=i
        
    query_index = index_number
    list=[]
    distance, indice = model_knn.kneighbors(wine_pivot.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)
    for i in range(0, len(distance.flatten())):
           list.append(wine_pivot.index[indice.flatten()[i]])
    return render_template("recommendation.html",data=list)

if __name__=="__main__":
    app.run(debug=True)