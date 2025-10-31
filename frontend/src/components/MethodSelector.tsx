import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';

interface Method {
  id: string;
  title: string;
  description: string;
}

interface MethodSelectorProps {
  title: string;
  subtitle: string;
  methods: Method[];
  onSelectMethod: (methodId: any) => void;
}

export function MethodSelector({ title, subtitle, methods, onSelectMethod }: MethodSelectorProps) {
  return (
    <div>
      <div className="text-center mb-8">
        <h2 className="text-3xl mb-2 text-white">{title}</h2>
        <p className="text-slate-400">{subtitle}</p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {methods.map((method) => (
          <Card
            key={method.id}
            className="bg-slate-800/50 border-slate-700 hover:border-purple-500 transition-all duration-300 hover:scale-105 cursor-pointer"
            onClick={() => onSelectMethod(method.id)}
          >
            <CardHeader>
              <CardTitle className="text-white">{method.title}</CardTitle>
              <CardDescription className="text-slate-400">
                {method.description}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700">
                Select Method
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
