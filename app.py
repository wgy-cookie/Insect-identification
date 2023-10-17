from flask import Flask, render_template, request, jsonify
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import torch.nn.functional as F
from flask_cors import CORS
import requests

# cors解决跨域问题
app = Flask(__name__)
CORS(app)


# 加载模型和定义数据预处理的函数
def load_model(model_path):
  transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

  if 'resnet' in model_path:
    model = torchvision.models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
  elif 'vgg' in model_path:
    model = torchvision.models.vgg16()
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 10)
  else:
    raise ValueError("Invalid model name")

  model.load_state_dict(torch.load(model_path))
  model.eval()

  return model, transform


# 预测结果处理函数
def process_prediction(outputs):
  # 处理输出结果
  probs = F.softmax(outputs, dim=1)
  return probs


def result(probs_vgg16, probs_resnet50):
  # 进行权重处理
  probs = probs_vgg16 * 0.7879234167893961 + probs_resnet50 * 0.2120765832106038
  ans = {
    0: '苹果蠹（读dù）蛾',
    1: '蝼蛄',
    2: '东方果实蝇',
    3: '菜青虫',
    4: '斜纹夜蛾',
    5: '象鼻虫',
    6: '叶蝉',
    7: '蝗虫',
    8: '蜗牛',
    9: '蝽'
  }
  # 根据模型预测结果，返回对应结果标签
  _, predicted = torch.max(probs, 1)
  predicted_index = predicted.item()
  predicted_prob = probs[0][predicted_index].item()
  predicted_class = ans[predicted_index]

  return predicted_class, round(predicted_prob, 4)


# html静态文件
@app.route('/')
def index():
  return render_template('Insect identification.html')


@app.route('/process_image', methods=['POST'])
def process_image():
  # 通过request获取前端传递过来的图片文件
  file = request.files.get('image')
  # 定义模型文件路径
  model_path_vgg16 = './vgg16_best_model.pth'
  model_path_resnet50 = './resnet50_best_model.pth'

  # 加载后的模型和图片处理

  model_vgg16, transform_vgg16 = load_model(model_path_vgg16)
  model_resnet50, transform_resnet50 = load_model(model_path_resnet50)

  # 对于前端传递过来的图片链接，通过request.get进行获取，并进行预处理
  if 'imageUrl' in request.form:
    imageUrl = request.form['imageUrl']
    response = requests.get(imageUrl, stream=True)
    response.raise_for_status()
    image = Image.open(response.raw)
  else:
    image = Image.open(file)

  # 对图片进行归一化处理
  image_vgg16 = transform_vgg16(image)
  image_vgg16 = image_vgg16.unsqueeze(0)  # 添加批次维度

  image_resnet50 = transform_resnet50(image)
  image_resnet50 = image_resnet50.unsqueeze(0)  # 添加批次维度

  # 使用VGG16模型进行分类识别
  with torch.no_grad():
    outputs_vgg16 = model_vgg16(image_vgg16)

  # 使用ResNet50模型进行分类识别
  with torch.no_grad():
    outputs_resnet50 = model_resnet50(image_resnet50)

  # 处理VGG16模型的预测结果
  probs_vgg16 = process_prediction(outputs_vgg16)

  # # 处理ResNet50模型的预测结果
  probs_resnet50 = process_prediction(outputs_resnet50)

  # 计算两个模型的权重结果
  name, algorithm_result = result(probs_vgg16, probs_resnet50)
  insect_introductions = {
    '苹果蠹（读dù）蛾': '苹果蠹（读dù）蛾（Cydia pomonella）是杂食性钻蛀害虫 ，属鳞翅目卷蛾科，有很强的适应性、抗逆性和繁殖能力，是一类对世界水果生产具有重大影响的有害生物。成虫体长8毫米，翅展15-22毫米，体灰褐色。前翅臀角处有深褐色椭圆形大斑，内有3条青铜色条纹，其间显出4-5条褐色横纹，翅基部颜色为浅灰色，中部颜色最浅，杂有波状纹。',
    '蝼蛄': '蛄，通常指蝼蛄，成虫体长30～35mm，灰褐色，腹部色较浅，全身密布细毛，头圆锥形，触角丝状，分布于全国各地。蛄食性杂，咬食植物根茎，果实，比如马铃薯，黄薯等等为农业害虫。前胸背板卵圆形，中间具一明显的暗红色长心脏形凹陷斑。前翅灰褐色，较短，仅达腹部中部。后翅扇形，较长，超过腹部末端。腹末具1对尾须。前足为开掘足，后足胫节背面内侧有4个距，别于华北蝼蛄。卵初产时长2.8mm，孵化前4mm，椭圆形，初产乳白色，后变黄褐色，孵化前暗紫色。有较强的趋光性，夜间活动，鸣叫。',
    '东方果实蝇': '东方果实蝇，原产于印度及马来半岛等地，为太平洋地区果树的重大害虫。其寄主繁多、繁殖力强，雌蝇产卵于果皮下，幼虫孵化后钻入果肉中蛀食，造成水果腐烂失去商品价值，如果吃水果时发现内部长虫，大半都是东方果实蝇的幼虫。由于台湾气候及环境适合它的生存，目前已遍布全台，终年均有果实蝇发生。',
    '菜青虫': '菜青虫，又称猪儿虫，是云贵川地区老家土话，它是一种长约五厘米的小虫，全身墨绿，肥敦敦胖乎乎，头部两根红色的须角。猪儿虫（又可称为“菜青虫”、“豆虫”），是昆虫纲中最常见的一目。大多栖居在鲜嫩的青菜叶上，还有水果树的嫩叶上，啃食植物嫩叶。是一种庸肥的食叶昆虫。',
    '斜纹夜蛾': '斜纹夜蛾，属鳞翅目夜蛾科斜纹夜蛾属的一个物种，是一种农作物害虫，褐色，前翅具许多斑纹，中有一条灰白色宽阔的斜纹，故名。中国除西藏、青海不详外，广泛分布于各地。寄主植物广泛，可危害各种农作物及观赏花木。斜纹夜蛾主要以幼虫危害，幼虫食性杂，且食量大，初孵幼虫在叶背为害，取食叶肉，仅留下表皮；3龄幼虫后造成叶片缺刻、残缺不堪甚至全部吃光，蚕食花蕾造成缺损，容易暴发成灾。幼虫体色变化很大，主要有3种：淡绿色、黑褐色、土黄色。',
    '象鼻虫': '象鼻虫是鞘翅目象甲科昆虫的通称。雄虫眼睛裸露、鼻子较短、体形较大；鞘翅边缘有绒毛，根据身体的大小、鼻子的长短很容易辨认雄雌；幼虫浅黄色，头部特别发达，能在植物茎内或谷物中蛀食。因喙像大象的长鼻子而得名。平均寿命为15-17天。只以雄花粉为食，幼虫以腐烂的花为食。天敌只有节高蜂和瘤节高蜂。有些种类的象鼻虫可不进行交配就能产卵繁衍后代，繁殖能力很强。',
    '叶蝉': '叶蝉科Cicadellidae 昆虫，隶属于半翅目（Hemiptera）头喙亚目 （Auchenorrhyncha）叶蝉总科（Cicadellidea）为一类害禾谷类、蔬菜、果树和林木等的昆虫。叶蝉作为半翅目最大的一个类群具有重要的经济意义。体长3-15毫米。单眼2个，少数种类无单眼。后足胫节有棱脊，棱脊上有3-4列刺状毛。后足胫节刺毛列是叶蝉科的最显著的识别特征。',
    '蝗虫': '蝗虫是蝗科蝗属的昆虫动物，全身通常为绿色、灰色、褐色或黑褐色，后腿肌肉发达，后足腿节粗壮，适于跳跃，强大的弹跳能力使它们成为跳跃能手。蝗虫分布于全世界的热带、温带的草地和沙漠地区。蝗虫是植食性昆虫，喜欢吃肥厚的叶子，有群居型和散居型之分，具迁飞性，蝗虫几乎都有典型的保护色，平时多数栖息于植物丛间。',
    '蜗牛': '蜗牛，是柄眼目蜗牛科的软体动物。蜗牛的身体柔软，外有一个螺旋形的外壳，躯体分头部和足部；头部有2对触角、后1对较长，后触角的顶端有1对眼，口腔内有颚及形似锉刀的齿舌，用来咀嚼及切碎食物；腹足扁平，底部分泌黏液，方便足部利用肌肉收缩，在不同表面上滑行，呼吸在类似肺的组织进行，空气由吸气孔进入。蜗牛喜欢在阴暗潮湿、疏松多腐殖质的环境中生活，昼伏夜出，最怕阳光直射，对环境反应敏感；眼睛的视力很差，在微弱光线下只能看6厘米远；触角嗅觉灵敏，靠触角、嗅觉寻找食物和配偶。蜗牛以植物性食物为主，尤其喜食蔬菜、果树的叶芽和作物的根叶。蜗牛为雌雄同体动物，但不能自体受精，交配一般都是在春秋季节，1-2个月后产卵。',
    '蝽': '蝽，半翅目昆虫，体小至中型体壁坚硬而体略扁平，刺吸式口器，着生于头的前端，不用时贴放在头胸的腹面。前胸背板发达，中胸有发达的小盾片。前翅基半部革质或角质，称为半鞘翅，一般分为革区、爪区和腹区三部分，有的种类有楔区。很多种类胸部腹面常有臭腺，可散发恶臭。多为植食性，刺吸植物茎叶或果实的液汁。'
  }
  data = {
    "insectName": name,
    "algorithmResult": algorithm_result,
    "insectIntroduction": insect_introductions.get(name)
  }

  return jsonify(data)


if __name__ == '__main__':
  app.run()
