loss, dout = softmax_loss(scores, y)
        for i in range(self.num_layers):
            loss += self.reg * np.sum(self.params['W' + str(i+1)]**2) / 2
        for i in range(self.num_layers, 0, -1):
            if i == self.num_layers:
                dout, grads['W'+ str(i+1)], grads['b' + str(i+1)] = affine_backward(dout, cache['cache' + str(i)])
            else:
                dout, grads['W'+ str(i+1)], grads['b' + str(i+1)] = affine_relu_backward(dout, cache['cache' + str(i)])
            grads['W'+ str(i)] += self.reg * np.sum(self.params['W' + str(i)])


out = {}
        cache = {}
        for i in range(self.num_layers):
            if i == 0:
                out['out' + str(i+1)], cache['cache' + str(i+1)] = affine_relu_forward(X, self.params['W1'], self.params['b1'])
            if i == self.num_layers - 1:
                out['out' + str(i+1)], cache['cache' + str(i+1)] = affine_forward(out['out' + str(i)], self.params['W' + str(i+1)], self.params['b' + str(i+1)])
            else:
                out['out' + str(i+1)], cache['cache' + str(i+1)] = affine_relu_forward(out['out' + str(i)], self.params['W' + str(i+1)], self.params['b' + str(i+1)])
        
        scores = out['out'+ str(self.num_layers)]