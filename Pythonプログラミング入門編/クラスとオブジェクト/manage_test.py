class Manage_test:
    def __init__(self, score):
        self.score = score
    
    def output_test_point(self):
        print(self.score)

class User:
    def __init__(self, user_id, username, email):
        self.user_id = user_id
        self.username = username
        self.email = email

    def get_userinfo(self):
        print(f'{self.username}/n{self.email}')

yuta = User(1234, 'Yuta Ito', 'yokosigo@gmail.com')
print(yuta.get_userinfo())
