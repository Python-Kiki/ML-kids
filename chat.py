import streamlit as st

# Set up the app title
st.title("Machine Learning Basics")


# Introduction
st.write("Hi there! Welcome to our Machine Learning Playground! 🚀")
st.write("Let's learn about different types of machine learning in a fun way!")

# Create a selectbox for the user to choose a machine learning type
ml_type = st.selectbox("Choose a Machine Learning Type:", ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"])

# Explanation and images for each type
if ml_type == "Supervised Learning":
    st.write("Supervised Learning is like teaching your computer with examples.")
    st.write("You show the computer pictures of cats and dogs and tell it which is which.")
    st.write("After learning, the computer can guess what's in new pictures!")
    st.write("Uses: Image recognition, language translation")
    st.image("https://image.freepik.com/free-vector/supervised-machine-learning-concept_24877-74373.jpg")

    supervised_method = st.selectbox("Choose a Supervised Learning Method:", ["Classification", "Regression"])

    if supervised_method == "Classification":
        st.write("In Classification, the computer learns to put things into different groups.")
        st.write("For example, it can learn to tell if an animal is a cat or a dog!")
        st.write("Uses: Spam email detection, disease diagnosis")
        st.image("https://image.freepik.com/free-vector/classification-machine-learning-concept_24877-74376.jpg")

        st.write("Demo Python Code:")
        st.code("""
                from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        # Load the dataset
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Create a classifier
        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(X_train, y_train)

        # Make predictions
        predictions = classifier.predict(X_test)
        print(predictions)
        """)

    elif supervised_method == "Regression":
        st.write("In Regression, the computer learns to predict numbers.")
        st.write("For example, it can learn to predict the price of a house based on its features!")
        st.write("Uses: Stock price prediction, weather forecasting")
        st.image("https://image.freepik.com/free-vector/regression-machine-learning-concept_24877-74377.jpg")

        st.write("Demo Python Code:")
        st.code("""
        from sklearn.linear_model import LinearRegression
        import numpy as np

        # Generate some random data
        X = np.random.rand(100, 1)
        y = 2 * X + 1 + np.random.rand(100, 1)

        # Create a regression model
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions
        new_X = np.array([[0.5]])
        predicted_y = model.predict(new_X)
        print(predicted_y)
        """)

elif ml_type == "Unsupervised Learning":
    st.write("Unsupervised Learning is when the computer looks for patterns by itself.")
    st.write("Imagine you have a bunch of colored marbles and want to group them.")
    st.write("The computer can help figure out which colors are similar!")
    st.write("Uses: Customer segmentation, recommendation systems")
    st.image("https://image.freepik.com/free-vector/unsupervised-machine-learning-concept_24877-74374.jpg")

    unsupervised_method = st.selectbox("Choose an Unsupervised Learning Method:", ["Clustering", "Dimensionality Reduction"])

    if unsupervised_method == "Clustering":
        st.write("In Clustering, the computer groups similar things together.")
        st.write("For example, it can group customers based on their shopping habits!")
        st.write("Uses: Market segmentation, social network analysis")
        st.image("https://image.freepik.com/free-vector/clustering-machine-learning-concept_24877-74378.jpg")

        st.write("Demo Python Code:")
        st.code("""
       from sklearn.cluster import KMeans
        import numpy as np

        # Generate some random data
        X = np.random.rand(100, 2)

        # Create a clustering model
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)

        # Get the cluster labels
        labels = kmeans.labels_
        print(labels)
        """)

    elif unsupervised_method == "Dimensionality Reduction":
        st.write("In Dimensionality Reduction, the computer simplifies data without losing too much information.")
        st.write("It's like making a summary of a long story!")
        st.write("Uses: Visualizing high-dimensional data, noise reduction")
        st.image("https://image.freepik.com/free-vector/dimensionality-reduction-machine-learning-concept_24877-74379.jpg")

        st.write("Demo Python Code:")
        st.code("""
        from sklearn.decomposition import PCA
        import numpy as np

        # Generate some random data
        X = np.random.rand(100, 5)

        # Create a PCA model
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        # Reduced data
        print(X_reduced)
        """)

# Explanation for real-time application
st.write("Let's see how machine learning is used in real life!")
st.write("Ever seen how YouTube recommends videos you might like?")
st.write("That's machine learning helping to show you things you'll enjoy!")

# Wrap-up
st.write("Isn't machine learning amazing? You can teach computers to be smart helpers!")
st.write("Remember, this is just the beginning. There's so much more to explore!")

# Add a fun image
st.image("https://image.freepik.com/free-vector/kids-studying-artificial-intelligence-robotics-cartoon-vector-illustration_114360-1289.jpg", caption="Learning together!")
st.subheader("By kiki")
st.markdown("[Follow me on Instagram](https://www.instagram.com/melmazinooo/)")
