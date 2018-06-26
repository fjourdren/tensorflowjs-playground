var tf = require('@tensorflow/tfjs');

class Model {

    constructor() {
        this.totalEpoch = 0
        this.model = this.generateModel()
    }

    generateModel() {
        let mod = tf.sequential();
        mod.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [1]}))
        mod.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [100]}))
        mod.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [100]}))
        mod.add(tf.layers.dense({units: 10, activation: 'relu', inputShape: [100]}))
        mod.add(tf.layers.dense({units: 1, activation: 'relu', inputShape: [10]}))

        mod.compile({ loss: 'meanSquaredError', optimizer: 'rmsprop' })

        return mod
    }

    async train(nbEpoch) {
        const training_data = tf.tensor([...Array(3).keys()], [3, 1])
        const target_data = tf.tensor([0.3,0.5,0.2], [3, 1])
        return new Promise((resolve, reject) => {
            this.model.fit(training_data, target_data, {
                epochs: nbEpoch,
                callbacks: {
                    onEpochEnd: async (epoch, log) => {
                        if(this.totalEpoch%10 === 0) {
                            console.log(`Epoch ${this.totalEpoch}: loss = ${log.loss}`)
                        }
                        this.totalEpoch++
                    }
                }
            }).then(fitOutput => {
                training_data.dispose()
                target_data.dispose()
                resolve(fitOutput)
            })

        })
    }

    predict(tensor) {
        return this.model.predict(tensor)
    }
}




async function go() {
    var modelInit = new Model()
     
    //total 500 epochs
    await modelInit.train(500)
    await modelInit.predict(tf.tensor([...Array(3).keys()], [3, 1])).print()
}

go()