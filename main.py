from pydantic import BaseModel


class Test(BaseModel):
    name: str

if __name__ == '__main__':
    print(type(Test(name="takeshi")))