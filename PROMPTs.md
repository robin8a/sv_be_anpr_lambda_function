# Prompts

## Lambda function

Help to create an aws lambda function with the following and name as "sv_be_anpr_lambda_function", for automatic number plate recognition

- Use the working @single-how-to-use-ultralytics-yolo-with-openai-for-number-plate-recognition.ipynb as template to create the function
- Recieves as parameters the src image s3 path, src path model "anpr-demo-mode.pt, google studio gemini api key.
- Return the plate number
- Use gemini
- Don't download the mode .pt, is going to be in s3 as a parameter
- Don't display the image


# Help to update Firebase Storage for the image

Help to update the lambda function having in count:

- The lambda function has already configured the following Environment variables:
sv_apiKey
sv_authDomain
sv_databaseURL
sv_projectId
sv_storageBucket
sv_messagingSenderId
sv_appId
sv_measurementId
bl_gemini_api_key
- To read the image use firebase storage using the necessary above variables to do it.
- Instead use "image_s3_uri" parameter use "image_firebase_key" for image store in google firebase and "sv_storageBucket" as a bucket
- Instead use "gemini_api_key" use the env variable "bl_gemini_api_key"
- Also update the @EventTest.json.example 

2. Using your current variables (Client SDK)If you prefer to stay with the variables you have, you will be using the standard firebase/storage package. Here is the flow your Lambda will follow
Steps: 

Initialize: Use initializeApp(firebaseConfig) with your variables.

Reference" Create a reference to the file path using ref(storage, 'path/to/image.jpg').

Download: Use getDownloadURL() to get a public link, or getBytes() to pull the data into the Lambda's memory.

Implementation Checklist for Lambda
Before you deploy, ensure you handle these three things:

Environment Variables: Do not hardcode those keys. Store them in AWS Lambda Environment Variables or AWS Secrets Manager.

Dependencies: Ensure your package.json includes firebase (for Client SDK) or firebase-admin (for Admin SDK).