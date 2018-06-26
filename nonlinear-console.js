var tf = require('@tensorflow/tfjs');
//var util = require('util')

async function go() {

    const model = tf.sequential();
    model.add(tf.layers.dense({units: 10, activation: 'relu', inputShape: [1]}))
    model.add(tf.layers.dense({units: 1, activation: 'relu'}))

    model.compile({ loss: 'meanSquaredError', optimizer: 'rmsprop' })

    const training_data = tf.tensor([[1], [2], [3]], [3, 1])
    const target_data = tf.tensor([0.3,0.5,0.2], [3, 1])

    for (let i = 1; i < 10 ; ++i) {
        var h = await model.fit(training_data, target_data, {epochs: 100});
        console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
    }

    //console.log(util.inspect(h, { maxArrayLength: null }));

    model.predict(training_data).print();

}

go();