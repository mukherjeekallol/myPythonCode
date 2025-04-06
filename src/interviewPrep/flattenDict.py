person = {
    "name":"kallol",
    "age":43,
    "homeAddress":{
        "city":"pittsburgh",
        "state":"pa"
    },
    "workAddress":{
        "city":"pittsburgh",
        "state":"pa",
        "bldg":{
            "dept":"engg",
            "floor":32
        }
    }
}

def flattenDict(prefix,input_d):
    output_d = {}
    for key in input_d:
        val = input_d[key]
        flat_key = prefix + key
        print(type(val))
        if type(val)==dict:
            output_d.update(flattenDict(flat_key+"_",val))
        else:
            output_d[flat_key] = val

    return output_d


print(flattenDict("",person))


