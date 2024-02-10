Welcome to this repository! This is the solution repo for the Chimp beauty :  Machine learning assignment. Firstly. to get everything setup, 
 
```
pip install requirements
```

To get recommendations for a user, run ```main.py``` with the username (string) as a parameter. 
This should return an array of 10 product codes that can be suggested to the user. 

**How does it work?**

<ul>
    <li> Similarity measures! We use the Product Brand and the Product type as parameters.
    <li> Cosine similarity measures are used to check for products that are similar to the ones that the users have ordered before.
    <li> This is made with the assumption that users would like to try similar product types that they need and brands that they trust.
    <li> The output (suggestions array) is truncated to 10 elements, so that we do not overload our users with suggestions.
    <li> Using product tags tocheck for similarity measures was also tested but it seemed to bring a lot of unpredictability to the output suggestions. However, with some more time and better data engineering my intuition is that it should add a lot of value as well!
</ul>
