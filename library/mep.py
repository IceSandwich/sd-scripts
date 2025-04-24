import json
from library.cipher import Cipher, NewCipher
import io
from PIL import Image
import typing

FileSuffix = ".mep"
MetadataFilename = "md" + FileSuffix

key: str = None
cipher: Cipher = None

def Init(mep_key: typing.Optional[str]):
	global key
	global cipher

	key = mep_key
	if mep_key is not None:
		cipher = NewCipher("xor", key)
		print("【】】】】 Using cipher: {}".format(cipher.GetName()))
	else:
		print("【】】】】 Cipher is not used.")

def ReadJSON(filename: str):
	assert(cipher is not None)
	with open(filename, 'rb') as f:
		data = f.read()
		cipher.ChangeKey(key[::-1])
		data = cipher.Decrypt(data)
	ret = json.loads(data.decode('utf-8'))
	return ret

def ReadImage(filename: str):
	assert(cipher is not None)
	with open(filename, 'rb') as f:
		data = f.read()
		cipher.ChangeKey(key)
		data = cipher.Decrypt(data)
	buf = io.BytesIO(data)
	img = Image.open(buf)
	return img

def SaveImage(filename: str, output: str):
	assert(cipher is not None)
	with open(filename, 'rb') as f:
		data = f.read()
	cipher.ChangeKey(key)
	data = cipher.Encrypt(data)
	with open(output, 'wb') as f:
		f.write(data)

def SaveMeta(filename: str, data: typing.Dict[str, typing.Any]):
	assert(cipher is not None)
	with open(filename, 'wb') as f:
		jsc = json.dumps(data)
		cipher.ChangeKey(key[::-1])
		data = jsc.encode('utf-8')
		data = cipher.Encrypt(data)
		f.write(data)
